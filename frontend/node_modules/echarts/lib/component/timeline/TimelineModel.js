
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
import { __extends } from "tslib";
import ComponentModel from '../../model/Component.js';
import SeriesData from '../../data/SeriesData.js';
import { each, isObject, clone } from 'zrender/lib/core/util.js';
import { convertOptionIdName, getDataItemValue } from '../../util/model.js';
var TimelineModel = /** @class */function (_super) {
  __extends(TimelineModel, _super);
  function TimelineModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TimelineModel.type;
    _this.layoutMode = 'box';
    return _this;
  }
  /**
   * @override
   */
  TimelineModel.prototype.init = function (option, parentModel, ecModel) {
    this.mergeDefaultAndTheme(option, ecModel);
    this._initData();
  };
  /**
   * @override
   */
  TimelineModel.prototype.mergeOption = function (option) {
    _super.prototype.mergeOption.apply(this, arguments);
    this._initData();
  };
  TimelineModel.prototype.setCurrentIndex = function (currentIndex) {
    if (currentIndex == null) {
      currentIndex = this.option.currentIndex;
    }
    var count = this._data.count();
    if (this.option.loop) {
      currentIndex = (currentIndex % count + count) % count;
    } else {
      currentIndex >= count && (currentIndex = count - 1);
      currentIndex < 0 && (currentIndex = 0);
    }
    this.option.currentIndex = currentIndex;
  };
  /**
   * @return {number} currentIndex
   */
  TimelineModel.prototype.getCurrentIndex = function () {
    return this.option.currentIndex;
  };
  /**
   * @return {boolean}
   */
  TimelineModel.prototype.isIndexMax = function () {
    return this.getCurrentIndex() >= this._data.count() - 1;
  };
  /**
   * @param {boolean} state true: play, false: stop
   */
  TimelineModel.prototype.setPlayState = function (state) {
    this.option.autoPlay = !!state;
  };
  /**
   * @return {boolean} true: play, false: stop
   */
  TimelineModel.prototype.getPlayState = function () {
    return !!this.option.autoPlay;
  };
  /**
   * @private
   */
  TimelineModel.prototype._initData = function () {
    var thisOption = this.option;
    var dataArr = thisOption.data || [];
    var axisType = thisOption.axisType;
    var names = this._names = [];
    var processedDataArr;
    if (axisType === 'category') {
      processedDataArr = [];
      each(dataArr, function (item, index) {
        var value = convertOptionIdName(getDataItemValue(item), '');
        var newItem;
        if (isObject(item)) {
          newItem = clone(item);
          newItem.value = index;
        } else {
          newItem = index;
        }
        processedDataArr.push(newItem);
        names.push(value);
      });
    } else {
      processedDataArr = dataArr;
    }
    var dimType = {
      category: 'ordinal',
      time: 'time',
      value: 'number'
    }[axisType] || 'number';
    var data = this._data = new SeriesData([{
      name: 'value',
      type: dimType
    }], this);
    data.initData(processedDataArr, names);
  };
  TimelineModel.prototype.getData = function () {
    return this._data;
  };
  /**
   * @public
   * @return {Array.<string>} categoreis
   */
  TimelineModel.prototype.getCategories = function () {
    if (this.get('axisType') === 'category') {
      return this._names.slice();
    }
  };
  TimelineModel.type = 'timeline';
  /**
   * @protected
   */
  TimelineModel.defaultOption = {
    // zlevel: 0,                  // 一级层叠
    z: 4,
    show: true,
    axisType: 'time',
    realtime: true,
    left: '20%',
    top: null,
    right: '20%',
    bottom: 0,
    width: null,
    height: 40,
    padding: 5,
    controlPosition: 'left',
    autoPlay: false,
    rewind: false,
    loop: true,
    playInterval: 2000,
    currentIndex: 0,
    itemStyle: {},
    label: {
      color: '#000'
    },
    data: []
  };
  return TimelineModel;
}(ComponentModel);
export default TimelineModel;