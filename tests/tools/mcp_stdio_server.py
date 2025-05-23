from fastmcp import FastMCP
import math

mcp = FastMCP("Math Wizardry Server 🔢✨")

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

if __name__ == "__main__":
    mcp.run()