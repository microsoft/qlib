"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.removeSensor = exports.getSensor = exports.Sensors = void 0;
var _id = _interopRequireDefault(require("./id"));
var _sensors = require("./sensors");
var _constant = require("./constant");
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
/**
 * Created by hustcc on 18/6/9.
 * Contract: i@hust.cc
 */

/**
 * all the sensor objects.
 * sensor pool
 */
var Sensors = {};

/**
 * When destroy the sensor, remove it from the pool
 */
exports.Sensors = Sensors;
function clean(sensorId) {
  // exist, then remove from pool
  if (sensorId && Sensors[sensorId]) {
    delete Sensors[sensorId];
  }
}

/**
 * get one sensor
 * @param element
 * @returns {*}
 */
var getSensor = function getSensor(element) {
  var sensorId = element.getAttribute(_constant.SizeSensorId);

  // 1. if the sensor exists, then use it
  if (sensorId && Sensors[sensorId]) {
    return Sensors[sensorId];
  }

  // 2. not exist, then create one
  var newId = (0, _id["default"])();
  element.setAttribute(_constant.SizeSensorId, newId);
  var sensor = (0, _sensors.createSensor)(element, function () {
    return clean(newId);
  });
  // add sensor into pool
  Sensors[newId] = sensor;
  return sensor;
};

/**
 * 移除 sensor
 * @param sensor
 */
exports.getSensor = getSensor;
var removeSensor = function removeSensor(sensor) {
  var sensorId = sensor.element.getAttribute(_constant.SizeSensorId);
  // remove event, dom of the sensor used
  sensor.destroy();
  clean(sensorId);
};
exports.removeSensor = removeSensor;