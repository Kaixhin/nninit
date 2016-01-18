package = "nninit"
version = "scm-1"

source = {
  url = "git://github.com/Kaixhin/nninit",
  tag = "master"
}

description = {
  summary = "Weight initialisation schemes for Torch7 neural network modules",
  detailed = [[
                Weight initialisation schemes for Torch7 neural network modules.
  ]],
  homepage = "https://github.com/Kaixhin/nninit",
  license = "MIT"
}

dependencies = {
  "torch >= 7.0",
  "nn >= 1.0"
}

build = {
  type = "builtin",
  modules = {
    nninit = "nninit.lua"
  }
}
