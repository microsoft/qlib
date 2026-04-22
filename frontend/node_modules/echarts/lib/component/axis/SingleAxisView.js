
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
import AxisBuilder from './AxisBuilder.js';
import * as graphic from '../../util/graphic.js';
import * as singleAxisHelper from '../../coord/single/singleAxisHelper.js';
import AxisView from './AxisView.js';
import { rectCoordAxisBuildSplitArea, rectCoordAxisHandleRemove } from './axisSplitHelper.js';
var axisBuilderAttrs = ['axisLine', 'axisTickLabel', 'axisName'];
var selfBuilderAttrs = ['splitArea', 'splitLine'];
var SingleAxisView = /** @class */function (_super) {
  __extends(SingleAxisView, _super);
  function SingleAxisView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = SingleAxisView.type;
    _this.axisPointerClass = 'SingleAxisPointer';
    return _this;
  }
  SingleAxisView.prototype.render = function (axisModel, ecModel, api, payload) {
    var group = this.group;
    group.removeAll();
    var oldAxisGroup = this._axisGroup;
    this._axisGroup = new graphic.Group();
    var layout = singleAxisHelper.layout(axisModel);
    var axisBuilder = new AxisBuilder(axisModel, layout);
    zrUtil.each(axisBuilderAttrs, axisBuilder.add, axisBuilder);
    group.add(this._axisGroup);
    group.add(axisBuilder.getGroup());
    zrUtil.each(selfBuilderAttrs, function (name) {
      if (axisModel.get([name, 'show'])) {
        axisElementBuilders[name](this, this.group, this._axisGroup, axisModel);
      }
    }, this);
    graphic.groupTransition(oldAxisGroup, this._axisGroup, axisModel);
    _super.prototype.render.call(this, axisModel, ecModel, api, payload);
  };
  SingleAxisView.prototype.remove = function () {
    rectCoordAxisHandleRemove(this);
  };
  SingleAxisView.type = 'singleAxis';
  return SingleAxisView;
}(AxisView);
var axisElementBuilders = {
  splitLine: function (axisView, group, axisGroup, axisModel) {
    var axis = axisModel.axis;
    if (axis.scale.isBlank()) {
      return;
    }
    var splitLineModel = axisModel.getModel('splitLine');
    var lineStyleModel = splitLineModel.getModel('lineStyle');
    var lineColors = lineStyleModel.get('color');
    lineColors = lineColors instanceof Array ? lineColors : [lineColors];
    var lineWidth = lineStyleModel.get('width');
    var gridRect = axisModel.coordinateSystem.getRect();
    var isHorizontal = axis.isHorizontal();
    var splitLines = [];
    var lineCount = 0;
    var ticksCoords = axis.getTicksCoords({
      tickModel: splitLineModel
    });
    var p1 = [];
    var p2 = [];
    for (var i = 0; i < ticksCoords.length; ++i) {
      var tickCoord = axis.toGlobalCoord(ticksCoords[i].coord);
      if (isHorizontal) {
        p1[0] = tickCoord;
        p1[1] = gridRect.y;
        p2[0] = tickCoord;
        p2[1] = gridRect.y + gridRect.height;
      } else {
        p1[0] = gridRect.x;
        p1[1] = tickCoord;
        p2[0] = gridRect.x + gridRect.width;
        p2[1] = tickCoord;
      }
      var line = new graphic.Line({
        shape: {
          x1: p1[0],
          y1: p1[1],
          x2: p2[0],
          y2: p2[1]
        },
        silent: true
      });
      graphic.subPixelOptimizeLine(line.shape, lineWidth);
      var colorIndex = lineCount++ % lineColors.length;
      splitLines[colorIndex] = splitLines[colorIndex] || [];
      splitLines[colorIndex].push(line);
    }
    var lineStyle = lineStyleModel.getLineStyle(['color']);
    for (var i = 0; i < splitLines.length; ++i) {
      group.add(graphic.mergePath(splitLines[i], {
        style: zrUtil.defaults({
          stroke: lineColors[i % lineColors.length]
        }, lineStyle),
        silent: true
      }));
    }
  },
  splitArea: function (axisView, group, axisGroup, axisModel) {
    rectCoordAxisBuildSplitArea(axisView, axisGroup, axisModel, axisModel);
  }
};
export default SingleAxisView;