'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function isObject(value) {
  return {}.toString.call(value).includes('Object');
}

exports.default = isObject;
