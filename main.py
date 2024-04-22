import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def grad(self, x):
        return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, 200*x[1] - 200*x[0]**2])
    
    def hess(self, x):
        return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])
    
def cauchy_step(grad, hess, tR):
    grad_norm = np.linalg.norm(grad)
    if grad_norm == 0:
        return np.zeros_like(grad)
    
    scaled_grad = grad / grad_norm
    hessian_times_grad = np.dot(hess, grad)
    curvature = np.dot(grad, hessian_times_grad)
  
    if curvature <= 0:
        return -tR * scaled_grad
    
    tau = min(grad_norm**3 / (tR * curvature), 1)
    
    return -tau * tR * scaled_grad


def optimize(model, x0, tR, N):
    x = x0
    x_hist = [x]
    f_hist = [model(x)]
    for i in range(N):
        grad = model.grad(x)
        hess = model.hess(x)
        
        s = cauchy_step(grad, hess, tR)
        # Compute the improvement in prediction
        pred_reduction = -np.dot(s, grad) - 0.5 * np.dot(np.dot(s, hess), s) # f(xk) - f(xk + s) = -s^T * grad - 0.5 * s^T * H * s
        x_new = x + s
        actual_reduction = model(x) - model(x_new)
        
        # Adjust the trust-region radius
        rho = actual_reduction / pred_reduction if pred_reduction != 0 else 0
        if rho < 0.25:
            tR *= 0.25  # Decrease the radius
        elif rho > 0.75 and np.linalg.norm(s) == tR:
            tR *= 2.0  # Increase the radius
        
        x = x_new
        x_hist.append(x)
        f_hist.append(model(x))    
    
    return x_hist, f_hist


def main():
    model = Model()
    x0 = np.array([0.0, 1.0])
    tR = 0.25
    N = 3000
    x_hist, f_hist = optimize(model, x0, tR, N)
    
    print("x:", x_hist[-1])
    print("f(x):", f_hist[-1])

    x_hist = np.array(x_hist)
    plt.plot(x_hist[:, 0], x_hist[:, 1], 'o-')
    plt.xlabel("$x_1$")
    plt.ylabel('$x_2$')
    plt.title('Solution Progression')
    plt.show()

    x = np.linspace(-0.05, 1.05, 10)
    y = np.linspace(-0.05, 1.05, 10)
    X, Y = np.meshgrid(x, y)
    Z = model(np.array([X, Y]))
    plt.contourf(X, Y, Z, levels=1000)
    plt.plot(x_hist[:, 0], x_hist[:, 1], 'o-', color='red')
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'text.usetex': True})
    plt.xlabel("$x_1$")
    plt.ylabel('$x_2$', rotation = 0)
    plt.title('Cauchy steps for minimizing Rosenbrock function')
    plt.colorbar()
    plt.clim(0, 100)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.show()


if __name__ == "__main__":
    main()
