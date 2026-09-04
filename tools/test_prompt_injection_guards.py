from __future__ import annotations

import unittest
from pathlib import Path

from tools.adaptive_rag.requirements import UNTRUSTED_EVIDENCE_RULE as REQUIREMENT_GUARD
from tools.adaptive_rag.verifier import UNTRUSTED_EVIDENCE_RULE as VERIFIER_GUARD
from tools.qwen_plus_rag_pipeline import ENHANCEMENT_SYSTEM_PROMPT, GENERATION_SYSTEM_PROMPT


class PromptInjectionGuardTest(unittest.TestCase):
    def test_all_model_stages_treat_evidence_as_untrusted_data(self) -> None:
        prompts = (
            REQUIREMENT_GUARD,
            VERIFIER_GUARD,
            ENHANCEMENT_SYSTEM_PROMPT,
            GENERATION_SYSTEM_PROMPT,
            Path("tools/adaptive_rag/controller.py").read_text(encoding="utf-8"),
        )
        for prompt in prompts:
            normalized = prompt.lower()
            self.assertIn("untrusted", normalized)
            self.assertIn("instructions", normalized)
            self.assertIn("evidence", normalized)


if __name__ == "__main__":
    unittest.main()
