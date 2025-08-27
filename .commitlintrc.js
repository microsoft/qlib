module.exports = {
    extends: ["@commitlint/config-conventional"],
    rules: {
        // Configuration Format: [level, applicability, value]
        // level: Error level, usually expressed as a number:
        //     0 - disable rule
        //     1 - Warning (does not prevent commits)
        //     2 - Error (will block the commit)
        // applicability: the conditions under which the rule applies, commonly used values:
        //     “always” - always apply the rule
        //     “never” - never apply the rule
        // value: the specific value of the rule, e.g. a maximum length of 100.
        // Refs: https://commitlint.js.org/reference/rules-configuration.html
      "header-max-length": [2, "always", 100],
      "type-enum": [
        2,
        "always",
        ["build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test", "Release-As"]
      ]
    }
  };
