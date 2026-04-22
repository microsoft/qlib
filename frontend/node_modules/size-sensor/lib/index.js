"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.ver = exports.clear = exports.bind = void 0;
var _sensorPool = require("./sensorPool");
/**
 * Created by hustcc on 18/6/9.[高考时间]
 * Contract: i@hust.cc
 */

/**
 * bind an element with resize callback function
 * @param {*} element
 * @param {*} cb
 */
var bind = function bind(element, cb) {
  var sensor = (0, _sensorPool.getSensor)(element);

  // listen with callback
  sensor.bind(cb);

  // return unbind function
  return function () {
    sensor.unbind(cb);
  };
};

/**
 * clear all the listener and sensor of an element
 * @param element
 */
exports.bind = bind;
var clear = function clear(element) {
  var sensor = (0, _sensorPool.getSensor)(element);
  (0, _sensorPool.removeSensor)(sensor);
};
exports.clear = clear;
var ver = "1.0.2";
exports.ver = ver;