import json
import unittest
from unittest.mock import patch

import main


class LoyaltyFallbackTest(unittest.TestCase):
    def test_code_interpreter_failure_uses_tier_discount_fallback(self):
        with patch(
            "main.code_session",
            side_effect=RuntimeError("Simulated Code Interpreter outage"),
        ):
            result = main.calculate_loyalty_discount(
                order_subtotal=150,
                loyalty_tier="Gold",
                points_available=4250,
                product_category="standard",
            )

        if isinstance(result, str):
            result = json.loads(result)

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["calculation_scope"], "tier_discount_only")
        self.assertEqual(result["tier_discount_pct"], 10.0)
        self.assertEqual(result["tier_discount"], 15.0)
        self.assertEqual(result["final_total"], 135.0)
        self.assertEqual(result["points_redeemed"], 0)
        self.assertEqual(result["points_earned"], 0)
        self.assertEqual(result["remaining_points"], 4250)
        self.assertIn("unavailable", result["note"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
