
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
import { Rect } from '../../util/graphic.js';
import * as formatUtil from '../../util/format.js';
import * as layout from '../../util/layout.js';
import VisualMapping from '../../visual/VisualMapping.js';
import ComponentView from '../../view/Component.js';
var VisualMapView = /** @class */function (_super) {
  __extends(VisualMapView, _super);
  function VisualMapView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = VisualMapView.type;
    _this.autoPositionValues = {
      left: 1,
      right: 1,
      top: 1,
      bottom: 1
    };
    return _this;
  }
  VisualMapView.prototype.init = function (ecModel, api) {
    this.ecModel = ecModel;
    this.api = api;
  };
  /**
   * @protected
   */
  VisualMapView.prototype.render = function (visualMapModel, ecModel, api, payload // TODO: TYPE
  ) {
    this.visualMapModel = visualMapModel;
    if (visualMapModel.get('show') === false) {
      this.group.removeAll();
      return;
    }
    this.doRender(visualMapModel, ecModel, api, payload);
  };
  /**
   * @protected
   */
  VisualMapView.prototype.renderBackground = function (group) {
    var visualMapModel = this.visualMapModel;
    var padding = formatUtil.normalizeCssArray(visualMapModel.get('padding') || 0);
    var rect = group.getBoundingRect();
    group.add(new Rect({
      z2: -1,
      silent: true,
      shape: {
        x: rect.x - padding[3],
        y: rect.y - padding[0],
        width: rect.width + padding[3] + padding[1],
        height: rect.height + padding[0] + padding[2]
      },
      style: {
        fill: visualMapModel.get('backgroundColor'),
        stroke: visualMapModel.get('borderColor'),
        lineWidth: visualMapModel.get('borderWidth')
      }
    }));
  };
  /**
   * @protected
   * @param targetValue can be Infinity or -Infinity
   * @param visualCluster Only can be 'color' 'opacity' 'symbol' 'symbolSize'
   * @param opts
   * @param opts.forceState Specify state, instead of using getValueState method.
   * @param opts.convertOpacityToAlpha For color gradient in controller widget.
   * @return {*} Visual value.
   */
  VisualMapView.prototype.getControllerVisual = function (targetValue, visualCluster, opts) {
    opts = opts || {};
    var forceState = opts.forceState;
    var visualMapModel = this.visualMapModel;
    var visualObj = {};
    // Default values.
    if (visualCluster === 'color') {
      var defaultColor = visualMapModel.get('contentColor');
      visualObj.color = defaultColor;
    }
    function getter(key) {
      return visualObj[key];
    }
    function setter(key, value) {
      visualObj[key] = value;
    }
    var mappings = visualMapModel.controllerVisuals[forceState || visualMapModel.getValueState(targetValue)];
    var visualTypes = VisualMapping.prepareVisualTypes(mappings);
    zrUtil.each(visualTypes, function (type) {
      var visualMapping = mappings[type];
      if (opts.convertOpacityToAlpha && type === 'opacity') {
        type = 'colorAlpha';
        visualMapping = mappings.__alphaForOpacity;
      }
      if (VisualMapping.dependsOn(type, visualCluster)) {
        visualMapping && visualMapping.applyVisual(targetValue, getter, setter);
      }
    });
    return visualObj[visualCluster];
  };
  VisualMapView.prototype.positionGroup = function (group) {
    var model = this.visualMapModel;
    var api = this.api;
    layout.positionElement(group, model.getBoxLayoutParams(), {
      width: api.getWidth(),
      height: api.getHeight()
    });
  };
  VisualMapView.prototype.doRender = function (visualMapModel, ecModel, api, payload) {};
  VisualMapView.type = 'visualMap';
  return VisualMapView;
}(ComponentView);
export default VisualMapView;