#!/usr/bin/env python3
"""Read-only paired trace audit. All recall values are lexical proxies."""
import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qwen_plus_rag_pipeline import content_tokens, load_jsonl
from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget, prioritize_contexts
from tools.adaptive_rag.requirements import Requirement, RequirementPlan


def replay_budget(source: dict, result: dict) -> dict:
    """Replay recorded non-oracle plan/config; never re-run a language model."""
    raw = result['requirements']
    plan = RequirementPlan(
        answerability=raw['answerability'],
        requirements=tuple(Requirement(r['id'], r['requirement'], r['status'],
                                      tuple(r['citations']), r.get('search_query', ''))
                           for r in raw['requirements']),
        selected_citations=tuple(raw['selected_citations']),
        conflict_citations=tuple(raw['conflict_citations']), model='recorded-plan-replay',
        latency_ms=0.0, usage={}, request_id='', attempts=0)
    b = result['budget']
    decision = BudgetDecision(b['mode'], b['initial_chars'], b['maximum_chars'],
                             b['max_contexts'], b['max_contexts_per_document'], tuple(b['reasons']))
    current = DynamicEvidenceBudget().build(source['contexts'], plan=plan, decision=decision)
    return {'qid': result['qid'], 'old_rendered_chars': b['rendered_chars'],
            'old_expanded': b['expanded_to_maximum'], 'replayed': current.to_dict(),
            'generation_rerun': False}


def audit(sources: list[dict], results: list[dict]) -> dict:
    source = {r['qid']: r for r in sources}
    if len(source) != len(sources) or len({r['qid'] for r in results}) != len(results):
        raise ValueError('Duplicate qids')
    signatures = {r.get('run_signature') for r in results if r.get('run_signature')}
    if len(signatures) > 1:
        raise ValueError('Mixed run signatures')
    rows, changes, contracts = [], [], []
    counts = Counter()
    for index, r in enumerate(results):
        if index % 100 == 0:
            print(f'auditing {index}/{len(results)}', file=sys.stderr, flush=True)
        s = source[r['qid']]
        if r['question'] != s['question']:
            raise ValueError('Question mismatch: ' + r['qid'])
        if r.get('error') or not s.get('answer_facts') or s['question_type'] == 'info_not_found':
            continue
        refs = [set(content_tokens(f)) for f in s['answer_facts']]
        def score(text):
            tokens = set(content_tokens(text))
            return statistics.fmean(len(tokens & f) / len(f) if f else float(not tokens) for f in refs)
        ctx, selected = s['contexts'], r['selected_contexts']
        plan, budget = r['requirements'], r['budget']
        capped = prioritize_contexts(ctx, selected_citations=plan['selected_citations'],
                    conflict_citations=plan['conflict_citations'],
                    max_per_document=budget['max_contexts_per_document'])
        texts = {'full': ctx, 'document_cap': capped,
                 'context_cap': capped[:budget['max_contexts']], 'prompt': selected}
        v = {k: score('\n'.join(c['text'] for c in value)) for k, value in texts.items()}
        v.update(pre=score(r['pre_verification_generation']['answer']),
                 answer=score(r['generation']['answer']), java_reported=s.get('evidence_fact_token_recall'))
        originals = {c['citation_id']: c for c in ctx}
        cut = {c['citation_id'] for c in selected if c['citation_id'] in originals and
               len(c['text']) < len(originals[c['citation_id']]['text'])}
        required = set(plan['selected_citations'])
        absent = required - {c['citation_id'] for c in selected}
        counts['truncated_rows'] += bool(cut)
        counts['truncated_required_rows'] += bool(cut & required)
        counts['absent_required_rows'] += bool(absent)
        counts['deterministic_plan_rows'] += plan.get('model') == 'deterministic'
        for key in ['citation_normalized', 'sentence_citation_normalized', 'requirement_coverage_normalized']:
            counts[key] += bool(r['pre_verification_generation'].get(key))
        if cut & required or absent:
            contracts.append({'qid': r['qid'], 'truncated': sorted(cut & required), 'absent': sorted(absent)})
        if r['pre_verification_generation']['answer'] != r['generation']['answer']:
            changes.append({'qid': r['qid'], 'status': r['verification']['status'], 'pre': v['pre'], 'post': v['answer']})
        rows.append(dict(qid=r['qid'], mode=r['router']['mode'], **v))
    keys = ['java_reported', 'full', 'document_cap', 'context_cap', 'prompt', 'pre', 'answer']
    def means(group):
        return {k: {'n': len(a), 'mean': statistics.fmean(a) if a else None}
                for k in keys for a in [[r[k] for r in group if r[k] is not None]]}
    return {'counts': dict(counts), 'means': means(rows),
            'by_mode': {m: means([r for r in rows if r['mode'] == m]) for m in ['fast', 'quality', 'deep']},
            'contracts': contracts, 'verifier_changes': changes,
            'largest_window_losses': sorted(rows, key=lambda r: r['prompt'] - r['full'])[:10],
            'largest_generation_losses': sorted(rows, key=lambda r: r['pre'] - r['prompt'])[:10]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts', required=True, type=Path)
    p.add_argument('--answers', required=True, type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--inspect-qid', action='append', default=[])
    p.add_argument('--replay-budget-qid', action='append', default=[])
    p.add_argument('--summary-output', type=Path)
    a = p.parse_args()
    sources, answers = load_jsonl(a.contexts), load_jsonl(a.answers)
    if a.summary_output and a.summary_output.resolve() in {a.contexts.resolve(), a.answers.resolve()}:
        p.error('Summary output must not overwrite either input')
    if a.replay_budget_qid:
        source_by_id = {s['qid']: s for s in sources}
        output = [replay_budget(source_by_id[r['qid']], r) for r in answers
                  if r['qid'] in a.replay_budget_qid]
        print(json.dumps(output, indent=2))
    elif a.inspect_qid:
        result_by_id = {r['qid']: r for r in answers}
        for s in sources:
            if s['qid'] not in a.inspect_qid:
                continue
            r = result_by_id[s['qid']]
            present = {c['citation_id']: len(c['text']) for c in r['selected_contexts']}
            print(json.dumps({'qid': s['qid'], 'question': s['question'],
                'answer': r['generation']['answer'],
                'expected_documents': s['expected_doc_ids'],
                'gold_document_contexts': [dict(citation=c['citation_id'], text=c['text'],
                    prompt_chars=present.get(c['citation_id'], 0)) for c in s['contexts']
                    if c['doc_id'] in s['expected_doc_ids']]}, ensure_ascii=False, indent=2))
    else:
        if a.limit is not None:
            answers = answers[:a.limit]
        output = audit(sources, answers)
        output['note'] = 'Lexical proxy, not semantic support or factual accuracy. Historical answers are NOT regenerated.'
        output['inputs'] = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for name, path in [('contexts', a.contexts), ('answers', a.answers)]}
        output['run_signatures'] = sorted({r['run_signature'] for r in answers if r.get('run_signature')})
        print(json.dumps(output, indent=2))
    if a.summary_output:
        if a.inspect_qid and not a.replay_budget_qid:
            p.error('--summary-output is not supported with --inspect-qid')
        a.summary_output.parent.mkdir(parents=True, exist_ok=True)
        a.summary_output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
