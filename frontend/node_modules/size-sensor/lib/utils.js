"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.isIE = void 0;
/**
 * Created by hustcc on 18/6/22.
 * Contract: i@hust.cc
 */

/**
 * whether is ie, should do something special for ie
 * @returns {RegExpMatchArray | null}
 */
var isIE = function isIE() {
  return navigator.userAgent.match(/Trident/) || navigator.userAgent.match(/Edge/);
};
exports.isIE = isIE;