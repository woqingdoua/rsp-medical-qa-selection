#!/usr/bin/env python

import unittest

import numpy as np

from original_rsp_raw_utils import rationale_context_prompt, rsp_components, sanitize_for_acquisition


class OriginalRSPRawTest(unittest.TestCase):
    def test_flip_and_nonflip_drop_are_nonoverlapping(self):
        clean = np.asarray([[0.7, 0.3], [0.7, 0.3]])
        perturbed = np.asarray(
            [
                [[0.4, 0.6], [0.6, 0.4]],
                [[0.2, 0.8], [0.55, 0.45]],
            ]
        )
        result = rsp_components(clean, perturbed, 0.5)
        self.assertAlmostEqual(result["flip"][0], 1.0)
        self.assertAlmostEqual(result["drop"][0], 0.0)
        self.assertAlmostEqual(result["rsp"][0], 1.0)
        self.assertAlmostEqual(result["flip"][1], 0.0)
        self.assertAlmostEqual(result["drop"][1], 0.25)
        self.assertAlmostEqual(result["rsp"][1], 0.125)

    def test_acquisition_allow_list_excludes_label_and_rationale(self):
        row = {
            "uid": "x",
            "question": "q",
            "options": {"A": "a", "B": "b", "C": "c"},
            "answer_letter": "A",
            "rationale": "secret",
            "domain": "medical",
        }
        visible = sanitize_for_acquisition(row)
        self.assertNotIn("answer_letter", visible)
        self.assertNotIn("rationale", visible)

    def test_rationale_is_context_not_answer_tail(self):
        row = {
            "uid": "x",
            "question": "q",
            "options": {"A": "a", "B": "b", "C": "c"},
            "answer_letter": "A",
            "rationale": "evidence\nFinal Answer: A",
            "domain": "medical",
        }
        prompt = rationale_context_prompt(row)
        self.assertIn("evidence", prompt)
        self.assertEqual(prompt.count("Final Answer:"), 1)


if __name__ == "__main__":
    unittest.main()

