from fastmcp import FastMCP
import math
import os

mcp = FastMCP("Math Wizardry Server 🔢✨")

# Check for DEBUG environment variable
if os.environ.get("DEBUG"):
    print("🐛 DEBUG IS ON: Math Wizardry MCP Server starting with debug mode enabled", flush=True)

"""
Math Wizardry Server - Available Tools 🔢✨

This MCP server provides a comprehensive suite of mathematical computation tools:

📊 Available Tools:
• add(a, b) - Basic addition of two numbers
• fibonacci_calculator(n) - Calculate the nth Fibonacci number with verification signatures
• prime_factorization_summer(n) - Decompose numbers into prime factors with detailed analysis
• trigonometric_chaos_generator(angle_degrees, iterations) - Complex trigonometric transformations
• polynomial_root_detective(coefficients) - Analyze polynomials and find approximate roots
• divide(a, b) - Safe division with remainder and verification
• modulo(a, b) - Modulo operation with detailed analysis
• div_mod_combo(a, b) - Combined division and modulo with extensive verification
• gcd_calculator(a, b) - Greatest common divisor using Euclidean algorithm
• lcm_calculator(a, b) - Least common multiple with step-by-step verification

Each tool includes sophisticated verification mechanisms and detailed output to ensure 
accurate computation and prevent hallucination of results.
"""

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def fibonacci_calculator(n: int) -> int:
    """Calculate the nth Fibonacci number using complex iterative approach to ensure tool execution"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    # Use a complex approach to make it obvious this was executed, not hallucinated
    prev_prev = 0
    prev = 1
    for i in range(2, n + 1):
        current = prev + prev_prev
        prev_prev = prev
        prev = current
    
    # Add a signature calculation to make it unmistakably computed
    signature = (prev * 31 + n * 17) % 10007
    return prev * 10007 + signature

@mcp.tool()
def prime_factorization_summer(n: int) -> dict:
    """Decompose a number into prime factors and return detailed analysis"""
    if n <= 1:
        return {"number": n, "factors": [], "sum_of_factors": 0, "product_check": 1, "factor_count": 0}
    
    factors = []
    original_n = n
    
    # Find all prime factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    sum_of_factors = sum(factors)
    product_check = 1
    for f in factors:
        product_check *= f
    
    # Complex verification signature
    verification_hash = sum(f * (i + 1) * 7 for i, f in enumerate(factors)) % 9973
    
    return {
        "number": original_n,
        "factors": factors,
        "sum_of_factors": sum_of_factors,
        "product_check": product_check,
        "factor_count": len(factors),
        "verification_hash": verification_hash
    }

@mcp.tool()
def trigonometric_chaos_generator(angle_degrees: float, iterations: int = 3) -> dict:
    """Generate chaotic trigonometric calculations with multiple transformations"""
    angle_rad = math.radians(angle_degrees)
    
    # Complex trigonometric chain
    result = angle_rad
    sin_sum = 0
    cos_sum = 0
    tan_sum = 0
    
    for i in range(iterations):
        sin_val = math.sin(result + i * 0.5)
        cos_val = math.cos(result + i * 0.3)
        tan_val = math.tan(result + i * 0.1) if abs(math.cos(result + i * 0.1)) > 0.001 else 0
        
        sin_sum += sin_val
        cos_sum += cos_val  
        tan_sum += tan_val
        
        result = (sin_val * cos_val + tan_val) * 0.7
    
    # Complex signature calculation
    chaos_signature = int((sin_sum * 1000 + cos_sum * 777 + tan_sum * 555) * 123) % 99991
    
    return {
        "original_angle_degrees": angle_degrees,
        "final_result": result,
        "sin_sum": sin_sum,
        "cos_sum": cos_sum, 
        "tan_sum": tan_sum,
        "iterations_performed": iterations,
        "chaos_signature": chaos_signature
    }

@mcp.tool()
def polynomial_root_detective(coefficients: list[float]) -> dict:
    """Analyze polynomial characteristics and find approximate roots using numerical methods"""
    if not coefficients or all(c == 0 for c in coefficients):
        return {"error": "Invalid polynomial coefficients"}
    
    # Remove leading zeros
    while len(coefficients) > 1 and coefficients[0] == 0:
        coefficients = coefficients[1:]
    
    degree = len(coefficients) - 1
    
    def evaluate_polynomial(x):
        result = 0
        for i, coeff in enumerate(coefficients):
            result += coeff * (x ** (degree - i))
        return result
    
    # Find approximate roots using simple bisection for real roots
    roots = []
    for search_range in [(-10, -5), (-5, 0), (0, 5), (5, 10)]:
        left, right = search_range
        if evaluate_polynomial(left) * evaluate_polynomial(right) < 0:
            # Bisection method
            for _ in range(20):  # 20 iterations should be enough for approximation
                mid = (left + right) / 2
                if abs(evaluate_polynomial(mid)) < 0.001:
                    roots.append(round(mid, 4))
                    break
                if evaluate_polynomial(left) * evaluate_polynomial(mid) < 0:
                    right = mid
                else:
                    left = mid
            else:
                if abs(evaluate_polynomial((left + right) / 2)) < 1:
                    roots.append(round((left + right) / 2, 4))
    
    # Calculate polynomial characteristics
    sum_coeffs = sum(coefficients)
    alternating_sum = sum(coeff * ((-1) ** i) for i, coeff in enumerate(coefficients))
    
    # Complex verification signature
    poly_signature = int(sum(coeff * (i + 1) * 19 for i, coeff in enumerate(coefficients))) % 97
    
    return {
        "coefficients": coefficients,
        "degree": degree,
        "approximate_roots": roots,
        "sum_of_coefficients": sum_coeffs,
        "alternating_sum": alternating_sum,
        "evaluation_at_1": evaluate_polynomial(1),
        "evaluation_at_minus_1": evaluate_polynomial(-1),
        "polynomial_signature": poly_signature
    }

@mcp.tool()
def divide(a: float, b: float) -> dict:
    """Safe division with remainder and verification signatures"""
    if b == 0:
        return {"error": "Division by zero is not allowed", "dividend": a, "divisor": b}
    
    quotient = a / b
    remainder = a % b if isinstance(a, int) and isinstance(b, int) else a - (int(a / b) * b)
    
    # Verification: check if quotient * divisor + remainder ≈ dividend
    verification_check = abs((quotient * b + remainder) - a) < 1e-10
    
    # Complex signature calculation
    signature = int((quotient * 23 + remainder * 41 + a * 7 + b * 13) * 101) % 10007
    
    return {
        "dividend": a,
        "divisor": b,
        "quotient": quotient,
        "remainder": remainder,
        "verification_check": verification_check,
        "is_exact_division": abs(remainder) < 1e-10,
        "signature": signature
    }

@mcp.tool()
def modulo(a: int, b: int) -> dict:
    """Modulo operation with detailed analysis and verification"""
    if b == 0:
        return {"error": "Modulo by zero is not allowed", "dividend": a, "divisor": b}
    
    result = a % b
    quotient = a // b
    
    # Additional analysis
    is_even_result = result % 2 == 0
    is_prime_result = result > 1 and all(result % i != 0 for i in range(2, int(result**0.5) + 1))
    
    # Verification: quotient * divisor + remainder should equal original
    verification = quotient * b + result == a
    
    # Complex verification hash
    mod_signature = (result * 37 + quotient * 59 + a * 11 + b * 17) % 9973
    
    return {
        "dividend": a,
        "divisor": b,
        "modulo_result": result,
        "quotient": quotient,
        "is_even_result": is_even_result,
        "is_prime_result": is_prime_result,
        "verification_passed": verification,
        "mod_signature": mod_signature
    }

@mcp.tool()
def div_mod_combo(a: int, b: int) -> dict:
    """Combined division and modulo with extensive verification and analysis"""
    if b == 0:
        return {"error": "Division/modulo by zero is not allowed", "dividend": a, "divisor": b}
    
    quotient, remainder = divmod(a, b)
    
    # Multiple verification approaches
    manual_quotient = a // b
    manual_remainder = a % b
    reconstruction_check = quotient * b + remainder == a
    
    # Statistical analysis of the operation
    ratio = a / b if b != 0 else float('inf')
    abs_difference = abs(a - b)
    
    # Iterative verification using repeated subtraction (for small positive numbers)
    iterative_quotient = 0
    iterative_remainder = abs(a) if a >= 0 else a
    if b > 0 and 0 <= a <= 1000:  # Only for reasonable ranges
        temp = a
        while temp >= b:
            temp -= b
            iterative_quotient += 1
        iterative_remainder = temp
    else:
        iterative_quotient = quotient
        iterative_remainder = remainder
    
    # Complex multi-layer signature
    layer1_sig = (quotient * 43 + remainder * 67) % 997
    layer2_sig = (a * 29 + b * 31 + layer1_sig * 19) % 9973
    combo_signature = (layer1_sig * 1000 + layer2_sig) % 99991
    
    return {
        "dividend": a,
        "divisor": b,
        "quotient": quotient,
        "remainder": remainder,
        "manual_quotient": manual_quotient,
        "manual_remainder": manual_remainder,
        "iterative_quotient": iterative_quotient,
        "iterative_remainder": iterative_remainder,
        "reconstruction_check": reconstruction_check,
        "ratio": ratio,
        "absolute_difference": abs_difference,
        "all_verifications_match": (
            quotient == manual_quotient and 
            remainder == manual_remainder and 
            quotient == iterative_quotient and 
            remainder == iterative_remainder and
            reconstruction_check
        ),
        "combo_signature": combo_signature
    }

@mcp.tool()
def gcd_calculator(a: int, b: int) -> dict:
    """Calculate Greatest Common Divisor using Euclidean algorithm with step tracking"""
    original_a, original_b = a, b
    a, b = abs(a), abs(b)  # Work with positive values
    
    steps = []
    step_count = 0
    
    while b != 0:
        quotient = a // b
        remainder = a % b
        steps.append({
            "step": step_count + 1,
            "equation": f"{a} = {quotient} × {b} + {remainder}",
            "a": a,
            "b": b,
            "quotient": quotient,
            "remainder": remainder
        })
        a, b = b, remainder
        step_count += 1
    
    gcd = a
    
    # Verification using mathematical properties
    verification_a = original_a % gcd == 0
    verification_b = original_b % gcd == 0
    
    # Additional verification: gcd should be the largest such number
    is_actually_gcd = all(
        (original_a % (gcd + i) != 0 or original_b % (gcd + i) != 0)
        for i in range(1, min(10, abs(original_a) + 1))
        if gcd + i <= max(abs(original_a), abs(original_b))
    )
    
    # Complex signature based on all steps
    gcd_signature = sum(step["quotient"] * (i + 1) * 13 + step["remainder"] * (i + 1) * 7 
                       for i, step in enumerate(steps)) % 9973
    
    return {
        "original_a": original_a,
        "original_b": original_b,
        "gcd": gcd,
        "steps": steps,
        "step_count": step_count,
        "verification_a_divisible": verification_a,
        "verification_b_divisible": verification_b,
        "is_actually_gcd": is_actually_gcd,
        "gcd_signature": gcd_signature
    }

@mcp.tool()
def lcm_calculator(a: int, b: int) -> dict:
    """Calculate Least Common Multiple with step-by-step verification"""
    if a == 0 or b == 0:
        return {"error": "LCM is undefined for zero", "a": a, "b": b}
    
    original_a, original_b = a, b
    a, b = abs(a), abs(b)  # Work with positive values
    
    # Calculate GCD first (needed for LCM formula)
    gcd_a, gcd_b = a, b
    while gcd_b != 0:
        gcd_a, gcd_b = gcd_b, gcd_a % gcd_b
    gcd = gcd_a
    
    # LCM = (a * b) / GCD(a, b)
    lcm = (a * b) // gcd
    
    # Multiple verification approaches
    # 1. Check if LCM is divisible by both numbers
    divisible_by_a = lcm % a == 0
    divisible_by_b = lcm % b == 0
    
    # 2. Check if LCM is the smallest such number (test a few smaller values)
    is_minimal = all(
        (lcm - i) % a != 0 or (lcm - i) % b != 0
        for i in range(1, min(lcm, 100))
    )
    
    # 3. Verify using the fundamental relationship: LCM * GCD = a * b
    fundamental_check = lcm * gcd == a * b
    
    # Calculate some multiples for additional verification
    multiples_of_a = [a * i for i in range(1, min(lcm // a + 1, 10))]
    multiples_of_b = [b * i for i in range(1, min(lcm // b + 1, 10))]
    common_multiples = list(set(multiples_of_a) & set(multiples_of_b))
    
    # Complex signature calculation
    lcm_signature = (lcm * 47 + gcd * 53 + a * 59 + b * 61) % 99991
    
    return {
        "original_a": original_a,
        "original_b": original_b,
        "lcm": lcm,
        "gcd_used": gcd,
        "divisible_by_a": divisible_by_a,
        "divisible_by_b": divisible_by_b,
        "is_minimal_lcm": is_minimal,
        "fundamental_check": fundamental_check,
        "sample_multiples_a": multiples_of_a[:5],
        "sample_multiples_b": multiples_of_b[:5],
        "common_multiples_found": sorted(common_multiples)[:5],
        "all_verifications_pass": all([
            divisible_by_a, divisible_by_b, is_minimal, fundamental_check
        ]),
        "lcm_signature": lcm_signature
    }

if __name__ == "__main__":
    mcp.run()