
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
import env from 'zrender/lib/core/env.js';
import { DataFormatMixin } from '../../model/mixin/dataFormat.js';
import ComponentModel from '../../model/Component.js';
import { makeInner, defaultEmphasis } from '../../util/model.js';
import { createTooltipMarkup } from '../tooltip/tooltipMarkup.js';
function fillLabel(opt) {
  defaultEmphasis(opt, 'label', ['show']);
}
// { [componentType]: MarkerModel }
var inner = makeInner();
var MarkerModel = /** @class */function (_super) {
  __extends(MarkerModel, _super);
  function MarkerModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = MarkerModel.type;
    /**
     * If marker model is created by self from series
     */
    _this.createdBySelf = false;
    return _this;
  }
  /**
   * @overrite
   */
  MarkerModel.prototype.init = function (option, parentModel, ecModel) {
    if (process.env.NODE_ENV !== 'production') {
      if (this.type === 'marker') {
        throw new Error('Marker component is abstract component. Use markLine, markPoint, markArea instead.');
      }
    }
    this.mergeDefaultAndTheme(option, ecModel);
    this._mergeOption(option, ecModel, false, true);
  };
  MarkerModel.prototype.isAnimationEnabled = function () {
    if (env.node) {
      return false;
    }
    var hostSeries = this.__hostSeries;
    return this.getShallow('animation') && hostSeries && hostSeries.isAnimationEnabled();
  };
  /**
   * @overrite
   */
  MarkerModel.prototype.mergeOption = function (newOpt, ecModel) {
    this._mergeOption(newOpt, ecModel, false, false);
  };
  MarkerModel.prototype._mergeOption = function (newOpt, ecModel, createdBySelf, isInit) {
    var componentType = this.mainType;
    if (!createdBySelf) {
      ecModel.eachSeries(function (seriesModel) {
        // mainType can be markPoint, markLine, markArea
        var markerOpt = seriesModel.get(this.mainType, true);
        var markerModel = inner(seriesModel)[componentType];
        if (!markerOpt || !markerOpt.data) {
          inner(seriesModel)[componentType] = null;
          return;
        }
        if (!markerModel) {
          if (isInit) {
            // Default label emphasis `position` and `show`
            fillLabel(markerOpt);
          }
          zrUtil.each(markerOpt.data, function (item) {
            // FIXME Overwrite fillLabel method ?
            if (item instanceof Array) {
              fillLabel(item[0]);
              fillLabel(item[1]);
            } else {
              fillLabel(item);
            }
          });
          markerModel = this.createMarkerModelFromSeries(markerOpt, this, ecModel);
          // markerModel = new ImplementedMarkerModel(
          //     markerOpt, this, ecModel
          // );
          zrUtil.extend(markerModel, {
            mainType: this.mainType,
            // Use the same series index and name
            seriesIndex: seriesModel.seriesIndex,
            name: seriesModel.name,
            createdBySelf: true
          });
          markerModel.__hostSeries = seriesModel;
        } else {
          markerModel._mergeOption(markerOpt, ecModel, true);
        }
        inner(seriesModel)[componentType] = markerModel;
      }, this);
    }
  };
  MarkerModel.prototype.formatTooltip = function (dataIndex, multipleSeries, dataType) {
    var data = this.getData();
    var value = this.getRawValue(dataIndex);
    var itemName = data.getName(dataIndex);
    return createTooltipMarkup('section', {
      header: this.name,
      blocks: [createTooltipMarkup('nameValue', {
        name: itemName,
        value: value,
        noName: !itemName,
        noValue: value == null
      })]
    });
  };
  MarkerModel.prototype.getData = function () {
    return this._data;
  };
  MarkerModel.prototype.setData = function (data) {
    this._data = data;
  };
  MarkerModel.prototype.getDataParams = function (dataIndex, dataType) {
    var params = DataFormatMixin.prototype.getDataParams.call(this, dataIndex, dataType);
    var hostSeries = this.__hostSeries;
    if (hostSeries) {
      params.seriesId = hostSeries.id;
      params.seriesName = hostSeries.name;
      params.seriesType = hostSeries.subType;
    }
    return params;
  };
  MarkerModel.getMarkerModelFromSeries = function (seriesModel,
  // Support three types of markers. Strict check.
  componentType) {
    return inner(seriesModel)[componentType];
  };
  MarkerModel.type = 'marker';
  MarkerModel.dependencies = ['series', 'grid', 'polar', 'geo'];
  return MarkerModel;
}(ComponentModel);
zrUtil.mixin(MarkerModel, DataFormatMixin.prototype);
export default MarkerModel;