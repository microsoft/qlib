
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
import { SOURCE_FORMAT_ARRAY_ROWS, SOURCE_FORMAT_OBJECT_ROWS } from '../../util/types.js';
import { makePrintable, throwError } from '../../util/log.js';
import { each } from 'zrender/lib/core/util.js';
import { normalizeToArray } from '../../util/model.js';
import { getRawValueParser, SortOrderComparator } from '../../data/helper/dataValueHelper.js';
var sampleLog = '';
if (process.env.NODE_ENV !== 'production') {
  sampleLog = ['Valid config is like:', '{ dimension: "age", order: "asc" }', 'or [{ dimension: "age", order: "asc"], { dimension: "date", order: "desc" }]'].join(' ');
}
export var sortTransform = {
  type: 'echarts:sort',
  transform: function (params) {
    var upstream = params.upstream;
    var config = params.config;
    var errMsg = '';
    // Normalize
    // const orderExprList: OrderExpression[] = isArray(config[0])
    //     ? config as OrderExpression[]
    //     : [config as OrderExpression];
    var orderExprList = normalizeToArray(config);
    if (!orderExprList.length) {
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'Empty `config` in sort transform.';
      }
      throwError(errMsg);
    }
    var orderDefList = [];
    each(orderExprList, function (orderExpr) {
      var dimLoose = orderExpr.dimension;
      var order = orderExpr.order;
      var parserName = orderExpr.parser;
      var incomparable = orderExpr.incomparable;
      if (dimLoose == null) {
        if (process.env.NODE_ENV !== 'production') {
          errMsg = 'Sort transform config must has "dimension" specified.' + sampleLog;
        }
        throwError(errMsg);
      }
      if (order !== 'asc' && order !== 'desc') {
        if (process.env.NODE_ENV !== 'production') {
          errMsg = 'Sort transform config must has "order" specified.' + sampleLog;
        }
        throwError(errMsg);
      }
      if (incomparable && incomparable !== 'min' && incomparable !== 'max') {
        var errMsg_1 = '';
        if (process.env.NODE_ENV !== 'production') {
          errMsg_1 = 'incomparable must be "min" or "max" rather than "' + incomparable + '".';
        }
        throwError(errMsg_1);
      }
      if (order !== 'asc' && order !== 'desc') {
        var errMsg_2 = '';
        if (process.env.NODE_ENV !== 'production') {
          errMsg_2 = 'order must be "asc" or "desc" rather than "' + order + '".';
        }
        throwError(errMsg_2);
      }
      var dimInfo = upstream.getDimensionInfo(dimLoose);
      if (!dimInfo) {
        if (process.env.NODE_ENV !== 'production') {
          errMsg = makePrintable('Can not find dimension info via: ' + dimLoose + '.\n', 'Existing dimensions: ', upstream.cloneAllDimensionInfo(), '.\n', 'Illegal config:', orderExpr, '.\n');
        }
        throwError(errMsg);
      }
      var parser = parserName ? getRawValueParser(parserName) : null;
      if (parserName && !parser) {
        if (process.env.NODE_ENV !== 'production') {
          errMsg = makePrintable('Invalid parser name ' + parserName + '.\n', 'Illegal config:', orderExpr, '.\n');
        }
        throwError(errMsg);
      }
      orderDefList.push({
        dimIdx: dimInfo.index,
        parser: parser,
        comparator: new SortOrderComparator(order, incomparable)
      });
    });
    // TODO: support it?
    var sourceFormat = upstream.sourceFormat;
    if (sourceFormat !== SOURCE_FORMAT_ARRAY_ROWS && sourceFormat !== SOURCE_FORMAT_OBJECT_ROWS) {
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'sourceFormat "' + sourceFormat + '" is not supported yet';
      }
      throwError(errMsg);
    }
    // Other upstream format are all array.
    var resultData = [];
    for (var i = 0, len = upstream.count(); i < len; i++) {
      resultData.push(upstream.getRawDataItem(i));
    }
    resultData.sort(function (item0, item1) {
      for (var i = 0; i < orderDefList.length; i++) {
        var orderDef = orderDefList[i];
        var val0 = upstream.retrieveValueFromItem(item0, orderDef.dimIdx);
        var val1 = upstream.retrieveValueFromItem(item1, orderDef.dimIdx);
        if (orderDef.parser) {
          val0 = orderDef.parser(val0);
          val1 = orderDef.parser(val1);
        }
        var result = orderDef.comparator.evaluate(val0, val1);
        if (result !== 0) {
          return result;
        }
      }
      return 0;
    });
    return {
      data: resultData
    };
  }
};