
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
import { parseConditionalExpression } from '../../util/conditionalExpression.js';
import { hasOwn, createHashMap } from 'zrender/lib/core/util.js';
import { makePrintable, throwError } from '../../util/log.js';
export var filterTransform = {
  type: 'echarts:filter',
  // PENDING: enhance to filter by index rather than create new data
  transform: function (params) {
    // [Caveat] Fail-Fast:
    // Do not return the whole dataset unless user config indicates it explicitly.
    // For example, if no condition is specified by mistake, returning an empty result
    // is better than returning the entire raw source for the user to find the mistake.
    var upstream = params.upstream;
    var rawItem;
    var condition = parseConditionalExpression(params.config, {
      valueGetterAttrMap: createHashMap({
        dimension: true
      }),
      prepareGetValue: function (exprOption) {
        var errMsg = '';
        var dimLoose = exprOption.dimension;
        if (!hasOwn(exprOption, 'dimension')) {
          if (process.env.NODE_ENV !== 'production') {
            errMsg = makePrintable('Relation condition must has prop "dimension" specified.', 'Illegal condition:', exprOption);
          }
          throwError(errMsg);
        }
        var dimInfo = upstream.getDimensionInfo(dimLoose);
        if (!dimInfo) {
          if (process.env.NODE_ENV !== 'production') {
            errMsg = makePrintable('Can not find dimension info via: ' + dimLoose + '.\n', 'Existing dimensions: ', upstream.cloneAllDimensionInfo(), '.\n', 'Illegal condition:', exprOption, '.\n');
          }
          throwError(errMsg);
        }
        return {
          dimIdx: dimInfo.index
        };
      },
      getValue: function (param) {
        return upstream.retrieveValueFromItem(rawItem, param.dimIdx);
      }
    });
    var resultData = [];
    for (var i = 0, len = upstream.count(); i < len; i++) {
      rawItem = upstream.getRawDataItem(i);
      if (condition.evaluate()) {
        resultData.push(rawItem);
      }
    }
    return {
      data: resultData
    };
  }
};