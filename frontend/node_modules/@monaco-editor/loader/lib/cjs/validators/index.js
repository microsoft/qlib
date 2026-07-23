'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var curry = require('../utils/curry.js');
var isObject = require('../utils/isObject.js');

/**
 * validates the configuration object and informs about deprecation
 * @param {Object} config - the configuration object 
 * @return {Object} config - the validated configuration object
 */
function validateConfig(config) {
  if (!config) errorHandler('configIsRequired');
  if (!isObject.default(config)) errorHandler('configType');
  if (config.urls) {
    informAboutDeprecation();
    return {
      paths: {
        vs: config.urls.monacoBase
      }
    };
  }
  return config;
}

/**
 * logs deprecation message
 */
function informAboutDeprecation() {
  console.warn(errorMessages.deprecation);
}
function throwError(errorMessages, type) {
  throw new Error(errorMessages[type] || errorMessages["default"]);
}
var errorMessages = {
  configIsRequired: 'the configuration object is required',
  configType: 'the configuration object should be an object',
  "default": 'an unknown error accured in `@monaco-editor/loader` package',
  deprecation: "Deprecation warning!\n    You are using deprecated way of configuration.\n\n    Instead of using\n      monaco.config({ urls: { monacoBase: '...' } })\n    use\n      monaco.config({ paths: { vs: '...' } })\n\n    For more please check the link https://github.com/suren-atoyan/monaco-loader#config\n  "
};
var errorHandler = curry.default(throwError)(errorMessages);
var validators = {
  config: validateConfig
};

exports.default = validators;
exports.errorHandler = errorHandler;
exports.errorMessages = errorMessages;
