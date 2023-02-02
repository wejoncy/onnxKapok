import sympy


def sympy_symbol(name):
    if isinstance(name, int):
        return sympy.Integer(name)
    elif isinstance(name, list):
        return [sympy_symbol(x) for x in name]
    return sympy.Symbol(name, integer=True, positive=True)


class IndexingDiv(sympy.Function):
    """
    a // b used in indexing where we need to be careful about simplification.
    We don't use sympy.FloorDiv to bypass some simplification rules.
    """

    nargs = (2,)

    @classmethod
    def eval(cls, base, divisor):
        if base == 0:
            return sympy.Integer(0)
        if divisor == 1:
            return base
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, IndexingDiv):
            return IndexingDiv(base.args[0], base.args[1] * divisor)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return IndexingDiv(base - a, divisor) + a / gcd
        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return IndexingDiv(
                sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
            )
