licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LowDiameterDecomposition",
    hdrs = ["LowDiameterDecomposition.h"],
    deps = [
        "//gbbs",
    ],
)

cc_binary(
    name = "LowDiameterDecomposition_main",
    srcs = ["LowDiameterDecomposition.cc"],
    deps = [":LowDiameterDecomposition"],
)
