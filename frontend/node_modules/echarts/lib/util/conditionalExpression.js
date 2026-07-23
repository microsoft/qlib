
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
import { keys, isArray, map, isObject, isString, isRegExp, isArrayLike, hasOwn, isNumber } from 'zrender/lib/core/util.js';
import { throwError, makePrintable } from './log.js';
import { getRawValueParser, createFilterComparator } from '../data/helper/dataValueHelper.js';
;
var RELATIONAL_EXPRESSION_OP_ALIAS_MAP = {
  value: 'eq',
  // PENDING: not good for literal semantic?
  '<': 'lt',
  '<=': 'lte',
  '>': 'gt',
  '>=': 'gte',
  '=': 'eq',
  '!=': 'ne',
  '<>': 'ne'
  // Might be misleading for sake of the difference between '==' and '===',
  // so don't support them.
  // '==': 'eq',
  // '===': 'seq',
  // '!==': 'sne'
  // PENDING: Whether support some common alias "ge", "le", "neq"?
  // ge: 'gte',
  // le: 'lte',
  // neq: 'ne',
};
// type RelationalExpressionOpEvaluate = (tarVal: unknown, condVal: unknown) => boolean;
var RegExpEvaluator = /** @class */function () {
  function RegExpEvaluator(rVal) {
    // Support condVal: RegExp | string
    var condValue = this._condVal = isString(rVal) ? new RegExp(rVal) : isRegExp(rVal) ? rVal : null;
    if (condValue == null) {
      var errMsg = '';
      if (process.env.NODE_ENV !== 'production') {
        errMsg = makePrintable('Illegal regexp', rVal, 'in');
      }
      throwError(errMsg);
    }
  }
  RegExpEvaluator.prototype.evaluate = function (lVal) {
    var type = typeof lVal;
    return isString(type) ? this._condVal.test(lVal) : isNumber(type) ? this._condVal.test(lVal + '') : false;
  };
  return RegExpEvaluator;
}();
var ConstConditionInternal = /** @class */function () {
  function ConstConditionInternal() {}
  ConstConditionInternal.prototype.evaluate = function () {
    return this.value;
  };
  return ConstConditionInternal;
}();
var AndConditionInternal = /** @class */function () {
  function AndConditionInternal() {}
  AndConditionInternal.prototype.evaluate = function () {
    var children = this.children;
    for (var i = 0; i < children.length; i++) {
      if (!children[i].evaluate()) {
        return false;
      }
    }
    return true;
  };
  return AndConditionInternal;
}();
var OrConditionInternal = /** @class */function () {
  function OrConditionInternal() {}
  OrConditionInternal.prototype.evaluate = function () {
    var children = this.children;
    for (var i = 0; i < children.length; i++) {
      if (children[i].evaluate()) {
        return true;
      }
    }
    return false;
  };
  return OrConditionInternal;
}();
var NotConditionInternal = /** @class */function () {
  function NotConditionInternal() {}
  NotConditionInternal.prototype.evaluate = function () {
    return !this.child.evaluate();
  };
  return NotConditionInternal;
}();
var RelationalConditionInternal = /** @class */function () {
  function RelationalConditionInternal() {}
  RelationalConditionInternal.prototype.evaluate = function () {
    var needParse = !!this.valueParser;
    // Call getValue with no `this`.
    var getValue = this.getValue;
    var tarValRaw = getValue(this.valueGetterParam);
    var tarValParsed = needParse ? this.valueParser(tarValRaw) : null;
    // Relational cond follow "and" logic internally.
    for (var i = 0; i < this.subCondList.length; i++) {
      if (!this.subCondList[i].evaluate(needParse ? tarValParsed : tarValRaw)) {
        return false;
      }
    }
    return true;
  };
  return RelationalConditionInternal;
}();
function parseOption(exprOption, getters) {
  if (exprOption === true || exprOption === false) {
    var cond = new ConstConditionInternal();
    cond.value = exprOption;
    return cond;
  }
  var errMsg = '';
  if (!isObjectNotArray(exprOption)) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = makePrintable('Illegal config. Expect a plain object but actually', exprOption);
    }
    throwError(errMsg);
  }
  if (exprOption.and) {
    return parseAndOrOption('and', exprOption, getters);
  } else if (exprOption.or) {
    return parseAndOrOption('or', exprOption, getters);
  } else if (exprOption.not) {
    return parseNotOption(exprOption, getters);
  }
  return parseRelationalOption(exprOption, getters);
}
function parseAndOrOption(op, exprOption, getters) {
  var subOptionArr = exprOption[op];
  var errMsg = '';
  if (process.env.NODE_ENV !== 'production') {
    errMsg = makePrintable('"and"/"or" condition should only be `' + op + ': [...]` and must not be empty array.', 'Illegal condition:', exprOption);
  }
  if (!isArray(subOptionArr)) {
    throwError(errMsg);
  }
  if (!subOptionArr.length) {
    throwError(errMsg);
  }
  var cond = op === 'and' ? new AndConditionInternal() : new OrConditionInternal();
  cond.children = map(subOptionArr, function (subOption) {
    return parseOption(subOption, getters);
  });
  if (!cond.children.length) {
    throwError(errMsg);
  }
  return cond;
}
function parseNotOption(exprOption, getters) {
  var subOption = exprOption.not;
  var errMsg = '';
  if (process.env.NODE_ENV !== 'production') {
    errMsg = makePrintable('"not" condition should only be `not: {}`.', 'Illegal condition:', exprOption);
  }
  if (!isObjectNotArray(subOption)) {
    throwError(errMsg);
  }
  var cond = new NotConditionInternal();
  cond.child = parseOption(subOption, getters);
  if (!cond.child) {
    throwError(errMsg);
  }
  return cond;
}
function parseRelationalOption(exprOption, getters) {
  var errMsg = '';
  var valueGetterParam = getters.prepareGetValue(exprOption);
  var subCondList = [];
  var exprKeys = keys(exprOption);
  var parserName = exprOption.parser;
  var valueParser = parserName ? getRawValueParser(parserName) : null;
  for (var i = 0; i < exprKeys.length; i++) {
    var keyRaw = exprKeys[i];
    if (keyRaw === 'parser' || getters.valueGetterAttrMap.get(keyRaw)) {
      continue;
    }
    var op = hasOwn(RELATIONAL_EXPRESSION_OP_ALIAS_MAP, keyRaw) ? RELATIONAL_EXPRESSION_OP_ALIAS_MAP[keyRaw] : keyRaw;
    var condValueRaw = exprOption[keyRaw];
    var condValueParsed = valueParser ? valueParser(condValueRaw) : condValueRaw;
    var evaluator = createFilterComparator(op, condValueParsed) || op === 'reg' && new RegExpEvaluator(condValueParsed);
    if (!evaluator) {
      if (process.env.NODE_ENV !== 'production') {
        errMsg = makePrintable('Illegal relational operation: "' + keyRaw + '" in condition:', exprOption);
      }
      throwError(errMsg);
    }
    subCondList.push(evaluator);
  }
  if (!subCondList.length) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = makePrintable('Relational condition must have at least one operator.', 'Illegal condition:', exprOption);
    }
    // No relational operator always disabled in case of dangers result.
    throwError(errMsg);
  }
  var cond = new RelationalConditionInternal();
  cond.valueGetterParam = valueGetterParam;
  cond.valueParser = valueParser;
  cond.getValue = getters.getValue;
  cond.subCondList = subCondList;
  return cond;
}
function isObjectNotArray(val) {
  return isObject(val) && !isArrayLike(val);
}
var ConditionalExpressionParsed = /** @class */function () {
  function ConditionalExpressionParsed(exprOption, getters) {
    this._cond = parseOption(exprOption, getters);
  }
  ConditionalExpressionParsed.prototype.evaluate = function () {
    return this._cond.evaluate();
  };
  return ConditionalExpressionParsed;
}();
;
export function parseConditionalExpression(exprOption, getters) {
  return new ConditionalExpressionParsed(exprOption, getters);
}