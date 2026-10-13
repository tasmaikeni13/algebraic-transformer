#!/usr/bin/env python3
"""Rational enclosures for the quintic Hermite weight and derivative errors.

Uses the classical C6 Hermite remainder and a centered Taylor bound for
its derivative. Radical endpoint intervals and the basis bounds use exact
rational arithmetic. This certifies real evaluation of the float32
coefficients; it does not certify the compiler or IEEE evaluation.
"""

from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.attention_interpolation_probe import coefficients


class Interval:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = F(lo), F(lo if hi is None else hi)
    def __add__(self, other):
        other = as_interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)
    __radd__ = __add__
    def __neg__(self):
        return Interval(-self.hi, -self.lo)
    def __sub__(self, other):
        return self + -as_interval(other)
    def __rsub__(self, other):
        return as_interval(other) + -self
    def __mul__(self, other):
        other = as_interval(other)
        values = [a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return Interval(min(values), max(values))
    __rmul__ = __mul__
    def __truediv__(self, other):
        other = as_interval(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("division interval contains zero")
        return self * Interval(1 / other.hi, 1 / other.lo)
    def __rtruediv__(self, other):
        return as_interval(other) / self
    def __pow__(self, power):
        if self.lo < 0 or power < 0:
            raise ValueError("positive powers only")
        return Interval(self.lo**power, self.hi**power)


def as_interval(x):
    return x if isinstance(x, Interval) else Interval(x)


def sqrt_interval(x, bits=90):
    x = F(x)
    scaled_floor = (x.numerator << (2*bits)) // x.denominator
    lower = math.isqrt(scaled_floor)
    return Interval(F(lower, 1 << bits), F(lower+1, 1 << bits))


def values(x):
    root = sqrt_interval(1+x*x)
    rho = root + x
    w = rho**16
    return w, 16*w/root, w*(256/(root*root) - 16*x/(root**3))


def derivative_polynomial(n, order):
    # f^(k)/f is a polynomial in x and r=(1+x^2)^(-1/2).
    # D(x^a r^b)=a*x^(a-1)*r^b - b*x^(a+1)*r^(b+2).
    terms = {(0,0): 1}
    for _ in range(order):
        next_terms = {}
        def add(pair, value):
            next_terms[pair] = next_terms.get(pair,0) + value
        for (a,b), value in terms.items():
            add((a,b+1), n*value)
            if a:
                add((a-1,b), a*value)
            if b:
                add((a+1,b+2), -b*value)
        terms = {k:v for k,v in next_terms.items() if v}
    return terms


def bernstein_min(poly, lo, hi):
    degree = len(poly)-1
    q = [sum(poly[k]*math.comb(k,j)*lo**(k-j)*(hi-lo)**j
             for k in range(j,degree+1)) for j in range(degree+1)]
    return min(sum(q[j]*F(math.comb(i,j),math.comb(degree,j)) for j in range(i+1))
               for i in range(degree+1))


def certify_basis():
    # Piecewise L1 sums of the value, first-derivative, and second-derivative
    # quintic Hermite basis derivatives, with endpoint sign factors explicit.
    pieces = [
        (F(15,4), [0,0,60,-120,60], F(0),F(1)),
        (F(1), [1,0,-6,4], F(0),F(1,3)),
        (F(1), [-1,0,30,-60,30], F(1,3),F(2,3)),
        (F(1), [-1,0,6,-4], F(2,3),F(1)),
        (F(1,10), [0,1,-3,2], F(0),F(2,5)),
        (F(1,10), [0,-1,6,-10,5], F(2,5),F(3,5)),
        (F(1,10), [0,-1,3,-2], F(3,5),F(1)),
    ]
    margins = []
    for bound, poly, lo, hi in pieces:
        gap = [-F(x) for x in poly]
        gap[0] += bound
        for i in range(64):
            a, b = lo+(hi-lo)*F(i,64), lo+(hi-lo)*F(i+1,64)
            margin = bernstein_min(gap,a,b)
            if margin < 0:
                raise ArithmeticError("basis certificate failed")
            margins.append(margin)
    return {"exact_rational_subintervals": len(margins), "bounds": ["15/4","1","1/10"],
            "minimum_bernstein_margin": float(min(margins)), "passed": True}


def main():
    coeff = coefficients(5)
    terms = derivative_polynomial(16,6)
    width = F(1,16)
    gamma_derivative = F(53,61440)
    rows = []
    for index in range(128):
        a, b = F(-4)+index*width, F(-4)+(index+1)*width
        w0,d0,dd0 = values(a)
        w1,d1,dd1 = values(b)
        c0,c1,c2 = w0, width*d0, width*width*dd0/2
        f,d,dd = w1-c0-c1-c2, width*d1-c1-2*c2, width*width*dd1-2*c2
        exact_coeff = [c0,c1,c2,10*f-4*d+dd/2,-15*f+7*d-dd,6*f-3*d+dd/2]
        coefficient_gaps = []
        for power, exact in enumerate(exact_coeff):
            literal = F.from_float(float(coeff[power,index]))
            coefficient_gaps.append(max(abs(literal-exact.lo),abs(literal-exact.hi)))
        min_abs = F(0) if a <= 0 <= b else min(abs(a),abs(b))
        max_abs = max(abs(a),abs(b))
        root_min,root_max = sqrt_interval(1+min_abs*min_abs),sqrt_interval(1+max_abs*max_abs)
        rmax,rmin = 1/root_min.lo,1/root_max.hi
        m6 = w1.hi*sum(abs(value)*max_abs**xp*rmax**rp for (xp,rp),value in terms.items())
        weight_error = m6*width**6/F(720*64) + sum(coefficient_gaps)
        derivative_error = m6*width**5*gamma_derivative + sum(i*e for i,e in enumerate(coefficient_gaps))/width
        rows.append({"index": index, "domain": [float(a),float(b)],
                     "weight_relative_bound": math.nextafter(float(weight_error/w0.lo),math.inf),
                     "derivative_relative_bound": math.nextafter(float(derivative_error/(16*rmin*w0.lo)),math.inf),
                     "coefficient_weight_relative_bound": math.nextafter(float(sum(coefficient_gaps)/w0.lo),math.inf)})
    report = {"schema": 1, "degree": 5, "domain": [-4,4], "bins": 128,
              "sqrt_enclosure_bits": 90, "arithmetic": "exact Python fractions until conservative float output rounding",
              "assumptions": ["classical C6 Hermite remainder with repeated endpoint nodes",
                              "centered degree-5 Taylor remainder for the derivative bound",
                              "analytic recurrence f^(k+1)/f = D(f^k/f)+16*r*(f^k/f)"],
              "basis_certificate": certify_basis(),
              "sixth_derivative_terms": [{"x_power":a,"r_power":b,"coefficient":v} for (a,b),v in terms.items()],
              "weight_max_relative_bound": max(x["weight_relative_bound"] for x in rows),
              "derivative_max_relative_bound": max(x["derivative_relative_bound"] for x in rows),
              "machine_evaluation_certified": False, "lean_imported": False, "intervals": rows}
    path = ROOT / "results/analysis/attention_interpolation_certificate.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["weight_max_relative_bound"],report["derivative_max_relative_bound"])


if __name__ == "__main__":
    main()
