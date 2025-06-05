import unittest
from qlib.utils.parser import parse_field
from qlib.data.base import Feature, PFeature, Expression
from qlib.data.ops import Add, Ref, Mean
from qlib.utils.parser import analyze_expression_raw_features

class TestParseField(unittest.TestCase):
    def test_simple_feature(self):
        self.assertEqual(parse_field("$close"), 'Feature("close")')
        self.assertEqual(parse_field("$open"), 'Feature("open")')
        self.assertEqual(parse_field("$volume5"), 'Feature("volume5")')

    def test_simple_pit_feature(self):
        self.assertEqual(parse_field("$$roe_q"), 'PFeature("roe_q")')
        self.assertEqual(parse_field("$$eps_a"), 'PFeature("eps_a")')

    def test_simple_operator(self):
        self.assertEqual(parse_field("Ref($close, 1)"), 'Operators.Ref(Feature("close"), 1)')
        self.assertEqual(parse_field("Mean($high, 5)"), 'Operators.Mean(Feature("high"), 5)')

    def test_nested_operator(self):
        self.assertEqual(parse_field("Add($low, Ref($close, -2))"), 'Operators.Add(Feature("low"), Operators.Ref(Feature("close"), -2))')

    def test_pit_feature_with_operator(self):
        self.assertEqual(parse_field("Ref($$roe_q, 1)"), 'Operators.Ref(PFeature("roe_q"), 1)')

    def test_multiple_features_and_operators(self):
        self.assertEqual(parse_field("Div(Sub($high, $low), $close)"), 'Operators.Div(Operators.Sub(Feature("high"), Feature("low")), Feature("close"))')
        self.assertEqual(parse_field("($high + $low) / 2"), '(Feature("high") + Feature("low")) / 2') # Note: parse_field only handles specific patterns

    def test_no_parsing_needed(self):
        self.assertEqual(parse_field("close"), 'close')
        self.assertEqual(parse_field("MyCustomFunction(val, 10)"), 'MyCustomFunction(val, 10)') # Assuming MyCustomFunction is not a registered Operator pattern

    def test_string_with_spaces_in_operator(self):
        self.assertEqual(parse_field("Ref ( $close , 1 ) "), 'Operators.Ref ( Feature("close") , 1 ) ')

    def test_case_sensitivity(self):
        self.assertEqual(parse_field("$CLOSE"), 'Feature("CLOSE")')
        self.assertEqual(parse_field("mean($volume, 5)"), 'Operators.mean(Feature("volume"), 5)') # Operator names are case sensitive based on regex


class TestAnalyzeExpressionDependencies(unittest.TestCase):
    def test_simple_feature_dependency(self):
        expr_obj = Feature("close") # Actual name stored is "close"
        self.assertEqual(analyze_expression_raw_features(expr_obj), {"$close"})

    def test_simple_pit_feature_dependency(self):
        expr_obj = PFeature("roe_q") # Actual name stored is "roe_q"
        self.assertEqual(analyze_expression_raw_features(expr_obj), {"$$roe_q"})

    def test_operator_dependency(self):
        close_f = Feature("close")
        high_f = Feature("high")
        expr_obj = Add(close_f, high_f)
        self.assertEqual(analyze_expression_raw_features(expr_obj), {"$close", "$high"})

    def test_nested_operator_dependency(self):
        close_f = Feature("close")
        open_f = Feature("open")
        volume_f = Feature("volume")
        expr_obj = Ref(Add(close_f, open_f), 5)
        self.assertEqual(analyze_expression_raw_features(expr_obj), {"$close", "$open"})
        
        expr_obj_2 = Mean(expr_obj, 10) # Mean(Ref(Add($close, $open), 5), 10)
        self.assertEqual(analyze_expression_raw_features(expr_obj_2), {"$close", "$open"})

    def test_mixed_features_dependency(self):
        close_f = Feature("close")
        roe_q_f = PFeature("roe_q")
        expr_obj = Add(close_f, Ref(roe_q_f, 1))
        self.assertEqual(analyze_expression_raw_features(expr_obj), {"$close", "$$roe_q"})

    def test_operator_with_constant(self):
        close_f = Feature("close")
        expr_obj = Add(close_f, 5) # 5 is not an Expression
        self.assertEqual(analyze_expression_raw_features(expr_obj), {"$close"})

    def test_no_feature_dependency(self):
        # This case is a bit artificial as operators usually take features.
        # If we had an operator that generates data without input features (e.g., a random number generator as an Expression)
        # Or an Expression that is just a constant (though not standard in Qlib's current ops)
        class ConstExpr(Expression):
            def get_required_raw_features(self) -> set: return super().get_required_raw_features()
            def _load_internal(self, instrument, start_index, end_index, *args): pass
            def get_longest_back_rolling(self): return 0
            def get_extended_window_size(self): return 0,0

        expr_obj = ConstExpr()
        self.assertEqual(analyze_expression_raw_features(expr_obj), set())

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            analyze_expression_raw_features("$close") # Should be an Expression object

if __name__ == '__main__':
    unittest.main() 