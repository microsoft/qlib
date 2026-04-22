
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
import * as zrUtil from 'zrender/lib/core/util.js';
import Model from '../../model/Model.js';
import { isNameSpecified } from '../../util/model.js';
import ComponentModel from '../../model/Component.js';
var getDefaultSelectorOptions = function (ecModel, type) {
  if (type === 'all') {
    return {
      type: 'all',
      title: ecModel.getLocaleModel().get(['legend', 'selector', 'all'])
    };
  } else if (type === 'inverse') {
    return {
      type: 'inverse',
      title: ecModel.getLocaleModel().get(['legend', 'selector', 'inverse'])
    };
  }
};
var LegendModel = /** @class */function (_super) {
  __extends(LegendModel, _super);
  function LegendModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = LegendModel.type;
    _this.layoutMode = {
      type: 'box',
      // legend.width/height are maxWidth/maxHeight actually,
      // whereas real width/height is calculated by its content.
      // (Setting {left: 10, right: 10} does not make sense).
      // So consider the case:
      // `setOption({legend: {left: 10});`
      // then `setOption({legend: {right: 10});`
      // The previous `left` should be cleared by setting `ignoreSize`.
      ignoreSize: true
    };
    return _this;
  }
  LegendModel.prototype.init = function (option, parentModel, ecModel) {
    this.mergeDefaultAndTheme(option, ecModel);
    option.selected = option.selected || {};
    this._updateSelector(option);
  };
  LegendModel.prototype.mergeOption = function (option, ecModel) {
    _super.prototype.mergeOption.call(this, option, ecModel);
    this._updateSelector(option);
  };
  LegendModel.prototype._updateSelector = function (option) {
    var selector = option.selector;
    var ecModel = this.ecModel;
    if (selector === true) {
      selector = option.selector = ['all', 'inverse'];
    }
    if (zrUtil.isArray(selector)) {
      zrUtil.each(selector, function (item, index) {
        zrUtil.isString(item) && (item = {
          type: item
        });
        selector[index] = zrUtil.merge(item, getDefaultSelectorOptions(ecModel, item.type));
      });
    }
  };
  LegendModel.prototype.optionUpdated = function () {
    this._updateData(this.ecModel);
    var legendData = this._data;
    // If selectedMode is single, try to select one
    if (legendData[0] && this.get('selectedMode') === 'single') {
      var hasSelected = false;
      // If has any selected in option.selected
      for (var i = 0; i < legendData.length; i++) {
        var name_1 = legendData[i].get('name');
        if (this.isSelected(name_1)) {
          // Force to unselect others
          this.select(name_1);
          hasSelected = true;
          break;
        }
      }
      // Try select the first if selectedMode is single
      !hasSelected && this.select(legendData[0].get('name'));
    }
  };
  LegendModel.prototype._updateData = function (ecModel) {
    var potentialData = [];
    var availableNames = [];
    ecModel.eachRawSeries(function (seriesModel) {
      var seriesName = seriesModel.name;
      availableNames.push(seriesName);
      var isPotential;
      if (seriesModel.legendVisualProvider) {
        var provider = seriesModel.legendVisualProvider;
        var names = provider.getAllNames();
        if (!ecModel.isSeriesFiltered(seriesModel)) {
          availableNames = availableNames.concat(names);
        }
        if (names.length) {
          potentialData = potentialData.concat(names);
        } else {
          isPotential = true;
        }
      } else {
        isPotential = true;
      }
      if (isPotential && isNameSpecified(seriesModel)) {
        potentialData.push(seriesModel.name);
      }
    });
    /**
     * @type {Array.<string>}
     * @private
     */
    this._availableNames = availableNames;
    // If legend.data is not specified in option, use availableNames as data,
    // which is convenient for user preparing option.
    var rawData = this.get('data') || potentialData;
    var legendNameMap = zrUtil.createHashMap();
    var legendData = zrUtil.map(rawData, function (dataItem) {
      // Can be string or number
      if (zrUtil.isString(dataItem) || zrUtil.isNumber(dataItem)) {
        dataItem = {
          name: dataItem
        };
      }
      if (legendNameMap.get(dataItem.name)) {
        // remove legend name duplicate
        return null;
      }
      legendNameMap.set(dataItem.name, true);
      return new Model(dataItem, this, this.ecModel);
    }, this);
    /**
     * @type {Array.<module:echarts/model/Model>}
     * @private
     */
    this._data = zrUtil.filter(legendData, function (item) {
      return !!item;
    });
  };
  LegendModel.prototype.getData = function () {
    return this._data;
  };
  LegendModel.prototype.select = function (name) {
    var selected = this.option.selected;
    var selectedMode = this.get('selectedMode');
    if (selectedMode === 'single') {
      var data = this._data;
      zrUtil.each(data, function (dataItem) {
        selected[dataItem.get('name')] = false;
      });
    }
    selected[name] = true;
  };
  LegendModel.prototype.unSelect = function (name) {
    if (this.get('selectedMode') !== 'single') {
      this.option.selected[name] = false;
    }
  };
  LegendModel.prototype.toggleSelected = function (name) {
    var selected = this.option.selected;
    // Default is true
    if (!selected.hasOwnProperty(name)) {
      selected[name] = true;
    }
    this[selected[name] ? 'unSelect' : 'select'](name);
  };
  LegendModel.prototype.allSelect = function () {
    var data = this._data;
    var selected = this.option.selected;
    zrUtil.each(data, function (dataItem) {
      selected[dataItem.get('name', true)] = true;
    });
  };
  LegendModel.prototype.inverseSelect = function () {
    var data = this._data;
    var selected = this.option.selected;
    zrUtil.each(data, function (dataItem) {
      var name = dataItem.get('name', true);
      // Initially, default value is true
      if (!selected.hasOwnProperty(name)) {
        selected[name] = true;
      }
      selected[name] = !selected[name];
    });
  };
  LegendModel.prototype.isSelected = function (name) {
    var selected = this.option.selected;
    return !(selected.hasOwnProperty(name) && !selected[name]) && zrUtil.indexOf(this._availableNames, name) >= 0;
  };
  LegendModel.prototype.getOrient = function () {
    return this.get('orient') === 'vertical' ? {
      index: 1,
      name: 'vertical'
    } : {
      index: 0,
      name: 'horizontal'
    };
  };
  LegendModel.type = 'legend.plain';
  LegendModel.dependencies = ['series'];
  LegendModel.defaultOption = {
    // zlevel: 0,
    z: 4,
    show: true,
    orient: 'horizontal',
    left: 'center',
    // right: 'center',
    top: 0,
    // bottom: null,
    align: 'auto',
    backgroundColor: 'rgba(0,0,0,0)',
    borderColor: '#ccc',
    borderRadius: 0,
    borderWidth: 0,
    padding: 5,
    itemGap: 10,
    itemWidth: 25,
    itemHeight: 14,
    symbolRotate: 'inherit',
    symbolKeepAspect: true,
    inactiveColor: '#ccc',
    inactiveBorderColor: '#ccc',
    inactiveBorderWidth: 'auto',
    itemStyle: {
      color: 'inherit',
      opacity: 'inherit',
      borderColor: 'inherit',
      borderWidth: 'auto',
      borderCap: 'inherit',
      borderJoin: 'inherit',
      borderDashOffset: 'inherit',
      borderMiterLimit: 'inherit'
    },
    lineStyle: {
      width: 'auto',
      color: 'inherit',
      inactiveColor: '#ccc',
      inactiveWidth: 2,
      opacity: 'inherit',
      type: 'inherit',
      cap: 'inherit',
      join: 'inherit',
      dashOffset: 'inherit',
      miterLimit: 'inherit'
    },
    textStyle: {
      color: '#333'
    },
    selectedMode: true,
    selector: false,
    selectorLabel: {
      show: true,
      borderRadius: 10,
      padding: [3, 5, 3, 5],
      fontSize: 12,
      fontFamily: 'sans-serif',
      color: '#666',
      borderWidth: 1,
      borderColor: '#666'
    },
    emphasis: {
      selectorLabel: {
        show: true,
        color: '#eee',
        backgroundColor: '#666'
      }
    },
    selectorPosition: 'auto',
    selectorItemGap: 7,
    selectorButtonGap: 10,
    tooltip: {
      show: false
    }
  };
  return LegendModel;
}(ComponentModel);
export default LegendModel;