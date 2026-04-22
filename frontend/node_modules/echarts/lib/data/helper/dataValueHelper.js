
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
import { parseDate, numericToNumber } from '../../util/number.js';
import { createHashMap, trim, hasOwn, isString, isNumber } from 'zrender/lib/core/util.js';
import { throwError } from '../../util/log.js';
/**
 * Convert raw the value in to inner value in List.
 *
 * [Performance sensitive]
 *
 * [Caution]: this is the key logic of user value parser.
 * For backward compatibility, do not modify it until you have to!
 */
export function parseDataValue(value,
// For high performance, do not omit the second param.
opt) {
  // Performance sensitive.
  var dimType = opt && opt.type;
  if (dimType === 'ordinal') {
    // If given value is a category string
    return value;
  }
  if (dimType === 'time'
  // spead up when using timestamp
  && !isNumber(value) && value != null && value !== '-') {
    value = +parseDate(value);
  }
  // dimType defaults 'number'.
  // If dimType is not ordinal and value is null or undefined or NaN or '-',
  // parse to NaN.
  // number-like string (like ' 123 ') can be converted to a number.
  // where null/undefined or other string will be converted to NaN.
  return value == null || value === '' ? NaN
  // If string (like '-'), using '+' parse to NaN
  // If object, also parse to NaN
  : Number(value);
}
;
var valueParserMap = createHashMap({
  'number': function (val) {
    // Do not use `numericToNumber` here. We have `numericToNumber` by default.
    // Here the number parser can have loose rule:
    // enable to cut suffix: "120px" => 120, "14%" => 14.
    return parseFloat(val);
  },
  'time': function (val) {
    // return timestamp.
    return +parseDate(val);
  },
  'trim': function (val) {
    return isString(val) ? trim(val) : val;
  }
});
export function getRawValueParser(type) {
  return valueParserMap.get(type);
}
var ORDER_COMPARISON_OP_MAP = {
  lt: function (lval, rval) {
    return lval < rval;
  },
  lte: function (lval, rval) {
    return lval <= rval;
  },
  gt: function (lval, rval) {
    return lval > rval;
  },
  gte: function (lval, rval) {
    return lval >= rval;
  }
};
var FilterOrderComparator = /** @class */function () {
  function FilterOrderComparator(op, rval) {
    if (!isNumber(rval)) {
      var errMsg = '';
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'rvalue of "<", ">", "<=", ">=" can only be number in filter.';
      }
      throwError(errMsg);
    }
    this._opFn = ORDER_COMPARISON_OP_MAP[op];
    this._rvalFloat = numericToNumber(rval);
  }
  // Performance sensitive.
  FilterOrderComparator.prototype.evaluate = function (lval) {
    // Most cases is 'number', and typeof maybe 10 times faseter than parseFloat.
    return isNumber(lval) ? this._opFn(lval, this._rvalFloat) : this._opFn(numericToNumber(lval), this._rvalFloat);
  };
  return FilterOrderComparator;
}();
var SortOrderComparator = /** @class */function () {
  /**
   * @param order by default: 'asc'
   * @param incomparable by default: Always on the tail.
   *        That is, if 'asc' => 'max', if 'desc' => 'min'
   *        See the definition of "incomparable" in [SORT_COMPARISON_RULE].
   */
  function SortOrderComparator(order, incomparable) {
    var isDesc = order === 'desc';
    this._resultLT = isDesc ? 1 : -1;
    if (incomparable == null) {
      incomparable = isDesc ? 'min' : 'max';
    }
    this._incomparable = incomparable === 'min' ? -Infinity : Infinity;
  }
  // See [SORT_COMPARISON_RULE].
  // Performance sensitive.
  SortOrderComparator.prototype.evaluate = function (lval, rval) {
    // Most cases is 'number', and typeof maybe 10 times faseter than parseFloat.
    var lvalFloat = isNumber(lval) ? lval : numericToNumber(lval);
    var rvalFloat = isNumber(rval) ? rval : numericToNumber(rval);
    var lvalNotNumeric = isNaN(lvalFloat);
    var rvalNotNumeric = isNaN(rvalFloat);
    if (lvalNotNumeric) {
      lvalFloat = this._incomparable;
    }
    if (rvalNotNumeric) {
      rvalFloat = this._incomparable;
    }
    if (lvalNotNumeric && rvalNotNumeric) {
      var lvalIsStr = isString(lval);
      var rvalIsStr = isString(rval);
      if (lvalIsStr) {
        lvalFloat = rvalIsStr ? lval : 0;
      }
      if (rvalIsStr) {
        rvalFloat = lvalIsStr ? rval : 0;
      }
    }
    return lvalFloat < rvalFloat ? this._resultLT : lvalFloat > rvalFloat ? -this._resultLT : 0;
  };
  return SortOrderComparator;
}();
export { SortOrderComparator };
var FilterEqualityComparator = /** @class */function () {
  function FilterEqualityComparator(isEq, rval) {
    this._rval = rval;
    this._isEQ = isEq;
    this._rvalTypeof = typeof rval;
    this._rvalFloat = numericToNumber(rval);
  }
  // Performance sensitive.
  FilterEqualityComparator.prototype.evaluate = function (lval) {
    var eqResult = lval === this._rval;
    if (!eqResult) {
      var lvalTypeof = typeof lval;
      if (lvalTypeof !== this._rvalTypeof && (lvalTypeof === 'number' || this._rvalTypeof === 'number')) {
        eqResult = numericToNumber(lval) === this._rvalFloat;
      }
    }
    return this._isEQ ? eqResult : !eqResult;
  };
  return FilterEqualityComparator;
}();
/**
 * [FILTER_COMPARISON_RULE]
 * `lt`|`lte`|`gt`|`gte`:
 * + rval must be a number. And lval will be converted to number (`numericToNumber`) to compare.
 * `eq`:
 * + If same type, compare with `===`.
 * + If there is one number, convert to number (`numericToNumber`) to compare.
 * + Else return `false`.
 * `ne`:
 * + Not `eq`.
 *
 *
 * [SORT_COMPARISON_RULE]
 * All the values are grouped into three categories:
 * + "numeric" (number and numeric string)
 * + "non-numeric-string" (string that excluding numeric string)
 * + "others"
 * "numeric" vs "numeric": values are ordered by number order.
 * "non-numeric-string" vs "non-numeric-string": values are ordered by ES spec (#sec-abstract-relational-comparison).
 * "others" vs "others": do not change order (always return 0).
 * "numeric" vs "non-numeric-string": "non-numeric-string" is treated as "incomparable".
 * "number" vs "others": "others" is treated as "incomparable".
 * "non-numeric-string" vs "others": "others" is treated as "incomparable".
 * "incomparable" will be seen as -Infinity or Infinity (depends on the settings).
 * MEMO:
 *   Non-numeric string sort makes sense when we need to put the items with the same tag together.
 *   But if we support string sort, we still need to avoid the misleading like `'2' > '12'`,
 *   So we treat "numeric-string" sorted by number order rather than string comparison.
 *
 *
 * [CHECK_LIST_OF_THE_RULE_DESIGN]
 * + Do not support string comparison until required. And also need to
 *   avoid the misleading of "2" > "12".
 * + Should avoid the misleading case:
 *   `" 22 " gte "22"` is `true` but `" 22 " eq "22"` is `false`.
 * + JS bad case should be avoided: null <= 0, [] <= 0, ' ' <= 0, ...
 * + Only "numeric" can be converted to comparable number, otherwise converted to NaN.
 *   See `util/number.ts#numericToNumber`.
 *
 * @return If `op` is not `RelationalOperator`, return null;
 */
export function createFilterComparator(op, rval) {
  return op === 'eq' || op === 'ne' ? new FilterEqualityComparator(op === 'eq', rval) : hasOwn(ORDER_COMPARISON_OP_MAP, op) ? new FilterOrderComparator(op, rval) : null;
}