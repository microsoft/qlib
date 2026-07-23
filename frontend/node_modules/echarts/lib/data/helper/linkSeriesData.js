
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
/**
 * Link lists and struct (graph or tree)
 */
import { curry, each, assert, extend, map, keys } from 'zrender/lib/core/util.js';
import { makeInner } from '../../util/model.js';
var inner = makeInner();
function linkSeriesData(opt) {
  var mainData = opt.mainData;
  var datas = opt.datas;
  if (!datas) {
    datas = {
      main: mainData
    };
    opt.datasAttr = {
      main: 'data'
    };
  }
  opt.datas = opt.mainData = null;
  linkAll(mainData, datas, opt);
  // Porxy data original methods.
  each(datas, function (data) {
    each(mainData.TRANSFERABLE_METHODS, function (methodName) {
      data.wrapMethod(methodName, curry(transferInjection, opt));
    });
  });
  // Beyond transfer, additional features should be added to `cloneShallow`.
  mainData.wrapMethod('cloneShallow', curry(cloneShallowInjection, opt));
  // Only mainData trigger change, because struct.update may trigger
  // another changable methods, which may bring about dead lock.
  each(mainData.CHANGABLE_METHODS, function (methodName) {
    mainData.wrapMethod(methodName, curry(changeInjection, opt));
  });
  // Make sure datas contains mainData.
  assert(datas[mainData.dataType] === mainData);
}
function transferInjection(opt, res) {
  if (isMainData(this)) {
    // Transfer datas to new main data.
    var datas = extend({}, inner(this).datas);
    datas[this.dataType] = res;
    linkAll(res, datas, opt);
  } else {
    // Modify the reference in main data to point newData.
    linkSingle(res, this.dataType, inner(this).mainData, opt);
  }
  return res;
}
function changeInjection(opt, res) {
  opt.struct && opt.struct.update();
  return res;
}
function cloneShallowInjection(opt, res) {
  // cloneShallow, which brings about some fragilities, may be inappropriate
  // to be exposed as an API. So for implementation simplicity we can make
  // the restriction that cloneShallow of not-mainData should not be invoked
  // outside, but only be invoked here.
  each(inner(res).datas, function (data, dataType) {
    data !== res && linkSingle(data.cloneShallow(), dataType, res, opt);
  });
  return res;
}
/**
 * Supplement method to List.
 *
 * @public
 * @param [dataType] If not specified, return mainData.
 */
function getLinkedData(dataType) {
  var mainData = inner(this).mainData;
  return dataType == null || mainData == null ? mainData : inner(mainData).datas[dataType];
}
/**
 * Get list of all linked data
 */
function getLinkedDataAll() {
  var mainData = inner(this).mainData;
  return mainData == null ? [{
    data: mainData
  }] : map(keys(inner(mainData).datas), function (type) {
    return {
      type: type,
      data: inner(mainData).datas[type]
    };
  });
}
function isMainData(data) {
  return inner(data).mainData === data;
}
function linkAll(mainData, datas, opt) {
  inner(mainData).datas = {};
  each(datas, function (data, dataType) {
    linkSingle(data, dataType, mainData, opt);
  });
}
function linkSingle(data, dataType, mainData, opt) {
  inner(mainData).datas[dataType] = data;
  inner(data).mainData = mainData;
  data.dataType = dataType;
  if (opt.struct) {
    data[opt.structAttr] = opt.struct;
    opt.struct[opt.datasAttr[dataType]] = data;
  }
  // Supplement method.
  data.getLinkedData = getLinkedData;
  data.getLinkedDataAll = getLinkedDataAll;
}
export default linkSeriesData;