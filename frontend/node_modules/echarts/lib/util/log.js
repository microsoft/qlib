
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/


/**
 * AUTO-GENERATED FILE. DO NOT MODIFY.
 */

/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
import { map, isString, isFunction, eqNaN, isRegExp } from 'zrender/lib/core/util.js';
var ECHARTS_PREFIX = '[ECharts] ';
var storedLogs = {};
var hasConsole = typeof console !== 'undefined'
// eslint-disable-next-line
&& console.warn && console.log;
function outputLog(type, str, onlyOnce) {
  if (hasConsole) {
    if (onlyOnce) {
      if (storedLogs[str]) {
        return;
      }
      storedLogs[str] = true;
    }
    // eslint-disable-next-line
    console[type](ECHARTS_PREFIX + str);
  }
}
export function log(str, onlyOnce) {
  outputLog('log', str, onlyOnce);
}
export function warn(str, onlyOnce) {
  outputLog('warn', str, onlyOnce);
}
export function error(str, onlyOnce) {
  outputLog('error', str, onlyOnce);
}
export function deprecateLog(str) {
  if (process.env.NODE_ENV !== 'production') {
    // Not display duplicate message.
    outputLog('warn', 'DEPRECATED: ' + str, true);
  }
}
export function deprecateReplaceLog(oldOpt, newOpt, scope) {
  if (process.env.NODE_ENV !== 'production') {
    deprecateLog((scope ? "[" + scope + "]" : '') + (oldOpt + " is deprecated, use " + newOpt + " instead."));
  }
}
/**
 * If in __DEV__ environment, get console printable message for users hint.
 * Parameters are separated by ' '.
 * @usage
 * makePrintable('This is an error on', someVar, someObj);
 *
 * @param hintInfo anything about the current execution context to hint users.
 * @throws Error
 */
export function makePrintable() {
  var hintInfo = [];
  for (var _i = 0; _i < arguments.length; _i++) {
    hintInfo[_i] = arguments[_i];
  }
  var msg = '';
  if (process.env.NODE_ENV !== 'production') {
    // Fuzzy stringify for print.
    // This code only exist in dev environment.
    var makePrintableStringIfPossible_1 = function (val) {
      return val === void 0 ? 'undefined' : val === Infinity ? 'Infinity' : val === -Infinity ? '-Infinity' : eqNaN(val) ? 'NaN' : val instanceof Date ? 'Date(' + val.toISOString() + ')' : isFunction(val) ? 'function () { ... }' : isRegExp(val) ? val + '' : null;
    };
    msg = map(hintInfo, function (arg) {
      if (isString(arg)) {
        // Print without quotation mark for some statement.
        return arg;
      } else {
        var printableStr = makePrintableStringIfPossible_1(arg);
        if (printableStr != null) {
          return printableStr;
        } else if (typeof JSON !== 'undefined' && JSON.stringify) {
          try {
            return JSON.stringify(arg, function (n, val) {
              var printableStr = makePrintableStringIfPossible_1(val);
              return printableStr == null ? val : printableStr;
            });
            // In most cases the info object is small, so do not line break.
          } catch (err) {
            return '?';
          }
        } else {
          return '?';
        }
      }
    }).join(' ');
  }
  return msg;
}
/**
 * @throws Error
 */
export function throwError(msg) {
  throw new Error(msg);
}