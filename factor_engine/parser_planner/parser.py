import re
from .ast import ASTNode, LeafNode, OperatorNode

class ExpressionParser:
    """
    一个简单的递归下降解析器，用于将因子表达式字符串转换为 AST。
    - 支持函数调用嵌套，例如: `add(rank(close), 1)`
    - 支持字面量（字段名或数字）作为参数。
    """

    def parse(self, expression: str) -> ASTNode:
        """
        解析给定的表达式字符串。
        """
        expression = expression.strip()

        # 尝试匹配 `func(args)` 模式
        match = re.match(r"(\w+)\s*\((.*)\)\s*$", expression)
        
        if match:
            # 是一个函数调用
            operator = match.group(1)
            args_str = match.group(2)
            
            if not args_str.strip():
                # 没有参数的函数，例如 `func()`
                return OperatorNode(operator, [])
                
            args = self._split_args(args_str)
            
            # 递归解析每个参数
            parsed_args = [self.parse(arg) for arg in args]
            return OperatorNode(operator, parsed_args)
        
        # 不是函数调用，应该是一个字面量 (e.g., 'close', '5', 'some_str')
        return self._parse_literal(expression)

    def _split_args(self, args_str: str) -> list[str]:
        """
        按逗号分割参数，同时正确处理括号嵌套。
        例如: "rank(close), ts_mean(high, 5)" -> ["rank(close)", "ts_mean(high, 5)"]
        """
        args = []
        start = 0
        balance = 0
        for i, char in enumerate(args_str):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            elif char == ',' and balance == 0:
                args.append(args_str[start:i].strip())
                start = i + 1
        
        # 添加最后一个参数
        args.append(args_str[start:].strip())
        return args

    def _parse_literal(self, literal_str: str) -> LeafNode:
        """
        解析一个字面量。
        - 如果是数字，则转换为 int 或 float。
        - 否则，视为字符串（例如字段名）。
        """
        literal_str = literal_str.strip()
        # 尝试转换为整数
        if re.fullmatch(r"-?\d+", literal_str):
            return LeafNode(int(literal_str))
        # 尝试转换为浮点数
        if re.fullmatch(r"-?\d+\.\d+", literal_str):
            return LeafNode(float(literal_str))
        # 视为字符串
        return LeafNode(literal_str)

# Example usage:
if __name__ == '__main__':
    parser = ExpressionParser()
    
    expr1 = "add(rank(close), rank(high))"
    ast1 = parser.parse(expr1)
    print(f"Expression: {expr1}")
    print(f"AST: {ast1}")
    # Expected: add('rank'('close'), 'rank'('high'))
    
    expr2 = "ts_mean(vwap, 10)"
    ast2 = parser.parse(expr2)
    print(f"Expression: {expr2}")
    print(f"AST: {ast2}")
    # Expected: ts_mean('vwap', '10')

    expr3 = "close"
    ast3 = parser.parse(expr3)
    print(f"Expression: {expr3}")
    print(f"AST: {ast3}")
    # Expected: 'close' 