from __future__ import annotations

import unittest
from tools.adaptive_rag.controller import GENERATION_SYSTEM_PROMPT as ADAPTIVE_GENERATION_PROMPT
from tools.adaptive_rag.requirements import REQUIREMENT_SYSTEM_PROMPT
from tools.adaptive_rag.verifier import VERIFY_SYSTEM_PROMPT
from tools.qwen_plus_rag_pipeline import ENHANCEMENT_SYSTEM_PROMPT, GENERATION_SYSTEM_PROMPT


class PromptInjectionGuardTest(unittest.TestCase):
    def test_all_model_stages_treat_evidence_as_untrusted_data(self) -> None:
        prompts = (
            REQUIREMENT_SYSTEM_PROMPT,
            VERIFY_SYSTEM_PROMPT,
            ENHANCEMENT_SYSTEM_PROMPT,
            GENERATION_SYSTEM_PROMPT,
            ADAPTIVE_GENERATION_PROMPT,
        )
        for prompt in prompts:
            normalized = prompt.lower()
            self.assertIn("untrusted", normalized)
            self.assertIn("instructions", normalized)
            self.assertIn("evidence", normalized)


if __name__ == "__main__":
    unittest.main()
