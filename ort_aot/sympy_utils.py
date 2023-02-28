import sympy
import re
from sympy.core.logic import fuzzy_and, fuzzy_or  # type: ignore[import]


def sympy_symbol(name):
    # if name == "sequence":
    #    #sequence is a keyword in sympy
    #    name = name+'_'
    if isinstance(name, int):
        return sympy.Integer(name)
    elif isinstance(name, list):
        return [sympy_symbol(x) for x in name]
    elif isinstance(name, str):
        if ' ' in name or ',' in name:
            name = re.sub(r'[^a-zA-Z0-9_]+', '_', name)
    return sympy.Symbol(name, integer=True, positive=True)


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
    """
    nargs = (2,)
    precedence = 50  # precedence of mul  # noqa: F811

    # Default return type for SymPy assumptions.
    # https://docs.sympy.org/latest/guides/assumptions.html#implementing-assumptions-handlers
    is_real = True

    @property
    def base(self):
        return self.args[0]

    @property
    def divisor(self):
        return self.args[1]

    def _sympystr(self, printer):
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f"{base}//{divisor}"

    # SymPy assumptions based on argument types.
    def _eval_is_real(self):
        return fuzzy_or([self.base.is_real, self.divisor.is_real])

    def _eval_is_integer(self):
        return fuzzy_and([self.base.is_integer, self.divisor.is_integer])

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base, divisor):
        def check_supported_type(x):
            if (x.is_integer is False and x.is_real is False and x.is_complex) or x.is_Boolean:
                raise TypeError(
                    f"unsupported operand type(s) for //: "
                    f"'{type(base).__name__}' and '{type(divisor).__name__}'"
                    f", expected integer or real")

        check_supported_type(base)
        check_supported_type(divisor)

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_real and divisor == 1:
            return sympy.floor(base)
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, (sympy.Integer, sympy.Float)) and isinstance(divisor, (sympy.Integer, sympy.Float)):
            return sympy.floor(base / divisor)
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return FloorDiv(base - a, divisor) + a / gcd

        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return FloorDiv(
                sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
            )
