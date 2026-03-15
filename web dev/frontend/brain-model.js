window.BRAIN_CONFIG = {
  classes: ["glioma", "meningioma", "pituitary", "no_tumor"],
  featureNames: [
    "mean_intensity",
    "std_intensity",
    "edge_density",
    "center_brightness",
    "symmetry",
    "entropy_proxy",
    "vertical_gradient",
    "horizontal_gradient",
  ],
  models: {
    logistic_regression: {
      type: "linear_softmax",
      coef: [
        [1.25, 0.45, 0.55, 0.75, -0.2, 0.25, 0.12, -0.08],
        [0.95, 0.68, 0.42, 0.55, 0.1, 0.35, -0.04, 0.11],
        [1.12, 0.5, 0.6, 0.8, -0.05, 0.22, 0.18, 0.05],
        [0.5, 0.25, 0.2, 0.3, 0.4, 0.12, -0.12, -0.1],
      ],
      intercept: [-0.9, -0.8, -0.85, -0.6],
    },
    svm: {
      type: "linear_softmax",
      coef: [
        [1.1, 0.52, 0.7, 0.66, -0.28, 0.18, 0.15, -0.04],
        [0.88, 0.74, 0.38, 0.6, 0.05, 0.4, -0.08, 0.2],
        [1.2, 0.48, 0.63, 0.77, -0.12, 0.24, 0.22, 0.03],
        [0.56, 0.32, 0.25, 0.26, 0.48, 0.18, -0.16, -0.12],
      ],
      intercept: [-0.86, -0.82, -0.84, -0.62],
      temperature: 0.9,
    },
    random_forest: {
      type: "rule_forest",
      rules: [
        { classIndex: 0, minCenter: 0.38, minEdge: 0.2, bonus: 0.2 },
        { classIndex: 1, minSym: 0.45, minStd: 0.2, bonus: 0.18 },
        { classIndex: 2, minMean: 0.4, minVert: 0.52, bonus: 0.22 },
        { classIndex: 3, maxEdge: 0.18, maxStd: 0.16, bonus: 0.3 },
      ],
      base: [0.25, 0.25, 0.25, 0.25],
    },
    cnn: {
      type: "nonlinear_softmax",
      coef: [
        [1.5, 0.8, 0.9, 1.1, -0.22, 0.3, 0.15, 0.1],
        [1.2, 0.95, 0.55, 0.75, 0.12, 0.42, -0.06, 0.18],
        [1.45, 0.7, 0.85, 1.2, -0.05, 0.24, 0.25, 0.08],
        [0.7, 0.38, 0.28, 0.35, 0.55, 0.16, -0.22, -0.18],
      ],
      intercept: [-1.2, -1.1, -1.15, -0.7],
      gamma: 1.35,
    },
    resnet50: {
      type: "nonlinear_softmax",
      coef: [
        [1.62, 0.85, 0.95, 1.22, -0.25, 0.34, 0.18, 0.13],
        [1.34, 1.03, 0.62, 0.88, 0.14, 0.48, -0.07, 0.24],
        [1.58, 0.76, 0.9, 1.28, -0.08, 0.3, 0.3, 0.1],
        [0.82, 0.42, 0.34, 0.43, 0.62, 0.19, -0.26, -0.2],
      ],
      intercept: [-1.26, -1.16, -1.2, -0.74],
      gamma: 1.5,
    },
  },
};
