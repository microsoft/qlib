
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
import * as modelUtil from '../../util/model.js';
import ComponentModel from '../../model/Component.js';
import Model from '../../model/Model.js';
import geoCreator from './geoCreator.js';
import geoSourceManager from './geoSourceManager.js';
;
var GeoModel = /** @class */function (_super) {
  __extends(GeoModel, _super);
  function GeoModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = GeoModel.type;
    return _this;
  }
  GeoModel.prototype.init = function (option, parentModel, ecModel) {
    var source = geoSourceManager.getGeoResource(option.map);
    if (source && source.type === 'geoJSON') {
      var itemStyle = option.itemStyle = option.itemStyle || {};
      if (!('color' in itemStyle)) {
        itemStyle.color = '#eee';
      }
    }
    this.mergeDefaultAndTheme(option, ecModel);
    // Default label emphasis `show`
    modelUtil.defaultEmphasis(option, 'label', ['show']);
  };
  GeoModel.prototype.optionUpdated = function () {
    var _this = this;
    var option = this.option;
    option.regions = geoCreator.getFilledRegions(option.regions, option.map, option.nameMap, option.nameProperty);
    var selectedMap = {};
    this._optionModelMap = zrUtil.reduce(option.regions || [], function (optionModelMap, regionOpt) {
      var regionName = regionOpt.name;
      if (regionName) {
        optionModelMap.set(regionName, new Model(regionOpt, _this, _this.ecModel));
        if (regionOpt.selected) {
          selectedMap[regionName] = true;
        }
      }
      return optionModelMap;
    }, zrUtil.createHashMap());
    if (!option.selectedMap) {
      option.selectedMap = selectedMap;
    }
  };
  /**
   * Get model of region.
   */
  GeoModel.prototype.getRegionModel = function (name) {
    return this._optionModelMap.get(name) || new Model(null, this, this.ecModel);
  };
  /**
   * Format label
   * @param name Region name
   */
  GeoModel.prototype.getFormattedLabel = function (name, status) {
    var regionModel = this.getRegionModel(name);
    var formatter = status === 'normal' ? regionModel.get(['label', 'formatter']) : regionModel.get(['emphasis', 'label', 'formatter']);
    var params = {
      name: name
    };
    if (zrUtil.isFunction(formatter)) {
      params.status = status;
      return formatter(params);
    } else if (zrUtil.isString(formatter)) {
      return formatter.replace('{a}', name != null ? name : '');
    }
  };
  GeoModel.prototype.setZoom = function (zoom) {
    this.option.zoom = zoom;
  };
  GeoModel.prototype.setCenter = function (center) {
    this.option.center = center;
  };
  // PENGING If selectedMode is null ?
  GeoModel.prototype.select = function (name) {
    var option = this.option;
    var selectedMode = option.selectedMode;
    if (!selectedMode) {
      return;
    }
    if (selectedMode !== 'multiple') {
      option.selectedMap = null;
    }
    var selectedMap = option.selectedMap || (option.selectedMap = {});
    selectedMap[name] = true;
  };
  GeoModel.prototype.unSelect = function (name) {
    var selectedMap = this.option.selectedMap;
    if (selectedMap) {
      selectedMap[name] = false;
    }
  };
  GeoModel.prototype.toggleSelected = function (name) {
    this[this.isSelected(name) ? 'unSelect' : 'select'](name);
  };
  GeoModel.prototype.isSelected = function (name) {
    var selectedMap = this.option.selectedMap;
    return !!(selectedMap && selectedMap[name]);
  };
  GeoModel.type = 'geo';
  GeoModel.layoutMode = 'box';
  GeoModel.defaultOption = {
    // zlevel: 0,
    z: 0,
    show: true,
    left: 'center',
    top: 'center',
    // Default value:
    // for geoSVG source: 1,
    // for geoJSON source: 0.75.
    aspectScale: null,
    // /// Layout with center and size
    // If you want to put map in a fixed size box with right aspect ratio
    // This two properties may be more convenient
    // layoutCenter: [50%, 50%]
    // layoutSize: 100
    silent: false,
    // Map type
    map: '',
    // Define left-top, right-bottom coords to control view
    // For example, [ [180, 90], [-180, -90] ]
    boundingCoords: null,
    // Default on center of map
    center: null,
    zoom: 1,
    scaleLimit: null,
    // selectedMode: false
    label: {
      show: false,
      color: '#000'
    },
    itemStyle: {
      borderWidth: 0.5,
      borderColor: '#444'
      // Default color:
      // + geoJSON: #eee
      // + geoSVG: null (use SVG original `fill`)
      // color: '#eee'
    },
    emphasis: {
      label: {
        show: true,
        color: 'rgb(100,0,0)'
      },
      itemStyle: {
        color: 'rgba(255,215,0,0.8)'
      }
    },
    select: {
      label: {
        show: true,
        color: 'rgb(100,0,0)'
      },
      itemStyle: {
        color: 'rgba(255,215,0,0.8)'
      }
    },
    regions: []
    // tooltip: {
    //     show: false
    // }
  };
  return GeoModel;
}(ComponentModel);
export default GeoModel;