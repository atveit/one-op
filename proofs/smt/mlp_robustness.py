import z3

def main():
    print("=== SMT Solver (Z3) Adversarial Robustness Verification ===")
    print("Verifying that a small L-infinity perturbation cannot flip the argmax of a 2D toy MLP layer.")

    solver = z3.Solver()

    # 1. Define the input variables (original embedding vector x)
    x1 = z3.Real('x1')
    x2 = z3.Real('x2')
    
    # Let's say the original, clean input token embedding is [1.0, -0.5]
    solver.add(x1 == 1.0)
    solver.add(x2 == -0.5)

    # 2. Define the adversarial perturbation variables (delta_x)
    dx1 = z3.Real('dx1')
    dx2 = z3.Real('dx2')
    
    # Bounded adversarial L-infinity norm (e.g., epsilon = 0.1)
    epsilon = 0.1
    solver.add(dx1 >= -epsilon, dx1 <= epsilon)
    solver.add(dx2 >= -epsilon, dx2 <= epsilon)

    # The perturbed input
    xp1 = x1 + dx1
    xp2 = x2 + dx2

    # 3. Define the network layer weights (W) and biases (b)
    # y = x * W + b
    # Let W = [[0.8, -0.2], 
    #          [0.3,  0.9]]
    # Let b = [0.0, 0.0]
    
    # Original output (y)
    y1 = x1 * 0.8 + x2 * 0.3
    y2 = x1 * -0.2 + x2 * 0.9

    # Perturbed output (y')
    yp1 = xp1 * 0.8 + xp2 * 0.3
    yp2 = xp1 * -0.2 + xp2 * 0.9

    # For the clean input, check which logit is larger:
    # y1 = (1.0 * 0.8) + (-0.5 * 0.3) = 0.8 - 0.15 = 0.65
    # y2 = (1.0 * -0.2) + (-0.5 * 0.9) = -0.2 - 0.45 = -0.65
    # So y1 > y2 natively. Argmax is index 0.

    # 4. The Adversarial Query
    # Can the adversary find a perturbation (dx1, dx2) within [-epsilon, epsilon] 
    # that flips the argmax so that yp2 >= yp1?
    adversarial_condition = (yp2 >= yp1)
    solver.add(adversarial_condition)

    print(f"\nChecking robustness for epsilon = {epsilon}...")
    result = solver.check()

    if result == z3.unsat:
        print("Result: UNSAT")
        print("Proof Successful! It is mathematically impossible for any adversarial perturbation within the epsilon ball to flip the model's prediction.")
    elif result == z3.sat:
        print("Result: SAT")
        print("Vulnerability Found! The solver generated an adversarial example:")
        m = solver.model()
        print(f"dx1 = {m[dx1].as_decimal(4)}, dx2 = {m[dx2].as_decimal(4)}")
    else:
        print("Result: UNKNOWN")

if __name__ == "__main__":
    main()