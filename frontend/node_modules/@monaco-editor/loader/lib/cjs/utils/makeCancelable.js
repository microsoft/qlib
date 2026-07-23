'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

// The source (has been changed) is https://github.com/facebook/react/issues/5465#issuecomment-157888325

var CANCELATION_MESSAGE = {
  type: 'cancelation',
  msg: 'operation is manually canceled'
};
function makeCancelable(promise) {
  var hasCanceled_ = false;
  var wrappedPromise = new Promise(function (resolve, reject) {
    promise.then(function (val) {
      return hasCanceled_ ? reject(CANCELATION_MESSAGE) : resolve(val);
    });
    promise["catch"](reject);
  });
  return wrappedPromise.cancel = function () {
    return hasCanceled_ = true;
  }, wrappedPromise;
}

exports.CANCELATION_MESSAGE = CANCELATION_MESSAGE;
exports.default = makeCancelable;
