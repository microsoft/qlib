"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;
/**
 * Created by hustcc on 18/6/9.
 * Contract: i@hust.cc
 */

var id = 1;

/**
 * generate unique id in application
 * @return {string}
 */
var _default = function _default() {
  return "".concat(id++);
};
exports["default"] = _default;