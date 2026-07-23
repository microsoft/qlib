
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
// TODO depends on DataZoom and Brush
import * as zrUtil from 'zrender/lib/core/util.js';
import BrushController from '../../helper/BrushController.js';
import BrushTargetManager from '../../helper/BrushTargetManager.js';
import * as history from '../../dataZoom/history.js';
import sliderMove from '../../helper/sliderMove.js';
import { ToolboxFeature } from '../featureManager.js';
import { makeInternalComponentId, parseFinder } from '../../../util/model.js';
import { registerInternalOptionCreator } from '../../../model/internalComponentCreator.js';
var each = zrUtil.each;
var DATA_ZOOM_ID_BASE = makeInternalComponentId('toolbox-dataZoom_');
var ICON_TYPES = ['zoom', 'back'];
var DataZoomFeature = /** @class */function (_super) {
  __extends(DataZoomFeature, _super);
  function DataZoomFeature() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  DataZoomFeature.prototype.render = function (featureModel, ecModel, api, payload) {
    if (!this._brushController) {
      this._brushController = new BrushController(api.getZr());
      this._brushController.on('brush', zrUtil.bind(this._onBrush, this)).mount();
    }
    updateZoomBtnStatus(featureModel, ecModel, this, payload, api);
    updateBackBtnStatus(featureModel, ecModel);
  };
  DataZoomFeature.prototype.onclick = function (ecModel, api, type) {
    handlers[type].call(this);
  };
  DataZoomFeature.prototype.remove = function (ecModel, api) {
    this._brushController && this._brushController.unmount();
  };
  DataZoomFeature.prototype.dispose = function (ecModel, api) {
    this._brushController && this._brushController.dispose();
  };
  DataZoomFeature.prototype._onBrush = function (eventParam) {
    var areas = eventParam.areas;
    if (!eventParam.isEnd || !areas.length) {
      return;
    }
    var snapshot = {};
    var ecModel = this.ecModel;
    this._brushController.updateCovers([]); // remove cover
    var brushTargetManager = new BrushTargetManager(makeAxisFinder(this.model), ecModel, {
      include: ['grid']
    });
    brushTargetManager.matchOutputRanges(areas, ecModel, function (area, coordRange, coordSys) {
      if (coordSys.type !== 'cartesian2d') {
        return;
      }
      var brushType = area.brushType;
      if (brushType === 'rect') {
        setBatch('x', coordSys, coordRange[0]);
        setBatch('y', coordSys, coordRange[1]);
      } else {
        setBatch({
          lineX: 'x',
          lineY: 'y'
        }[brushType], coordSys, coordRange);
      }
    });
    history.push(ecModel, snapshot);
    this._dispatchZoomAction(snapshot);
    function setBatch(dimName, coordSys, minMax) {
      var axis = coordSys.getAxis(dimName);
      var axisModel = axis.model;
      var dataZoomModel = findDataZoom(dimName, axisModel, ecModel);
      // Restrict range.
      var minMaxSpan = dataZoomModel.findRepresentativeAxisProxy(axisModel).getMinMaxSpan();
      if (minMaxSpan.minValueSpan != null || minMaxSpan.maxValueSpan != null) {
        minMax = sliderMove(0, minMax.slice(), axis.scale.getExtent(), 0, minMaxSpan.minValueSpan, minMaxSpan.maxValueSpan);
      }
      dataZoomModel && (snapshot[dataZoomModel.id] = {
        dataZoomId: dataZoomModel.id,
        startValue: minMax[0],
        endValue: minMax[1]
      });
    }
    function findDataZoom(dimName, axisModel, ecModel) {
      var found;
      ecModel.eachComponent({
        mainType: 'dataZoom',
        subType: 'select'
      }, function (dzModel) {
        var has = dzModel.getAxisModel(dimName, axisModel.componentIndex);
        has && (found = dzModel);
      });
      return found;
    }
  };
  ;
  DataZoomFeature.prototype._dispatchZoomAction = function (snapshot) {
    var batch = [];
    // Convert from hash map to array.
    each(snapshot, function (batchItem, dataZoomId) {
      batch.push(zrUtil.clone(batchItem));
    });
    batch.length && this.api.dispatchAction({
      type: 'dataZoom',
      from: this.uid,
      batch: batch
    });
  };
  DataZoomFeature.getDefaultOption = function (ecModel) {
    var defaultOption = {
      show: true,
      filterMode: 'filter',
      // Icon group
      icon: {
        zoom: 'M0,13.5h26.9 M13.5,26.9V0 M32.1,13.5H58V58H13.5 V32.1',
        back: 'M22,1.4L9.9,13.5l12.3,12.3 M10.3,13.5H54.9v44.6 H10.3v-26'
      },
      // `zoom`, `back`
      title: ecModel.getLocaleModel().get(['toolbox', 'dataZoom', 'title']),
      brushStyle: {
        borderWidth: 0,
        color: 'rgba(210,219,238,0.2)'
      }
    };
    return defaultOption;
  };
  return DataZoomFeature;
}(ToolboxFeature);
var handlers = {
  zoom: function () {
    var nextActive = !this._isZoomActive;
    this.api.dispatchAction({
      type: 'takeGlobalCursor',
      key: 'dataZoomSelect',
      dataZoomSelectActive: nextActive
    });
  },
  back: function () {
    this._dispatchZoomAction(history.pop(this.ecModel));
  }
};
function makeAxisFinder(dzFeatureModel) {
  var setting = {
    xAxisIndex: dzFeatureModel.get('xAxisIndex', true),
    yAxisIndex: dzFeatureModel.get('yAxisIndex', true),
    xAxisId: dzFeatureModel.get('xAxisId', true),
    yAxisId: dzFeatureModel.get('yAxisId', true)
  };
  // If both `xAxisIndex` `xAxisId` not set, it means 'all'.
  // If both `yAxisIndex` `yAxisId` not set, it means 'all'.
  // Some old cases set like this below to close yAxis control but leave xAxis control:
  // `{ feature: { dataZoom: { yAxisIndex: false } }`.
  if (setting.xAxisIndex == null && setting.xAxisId == null) {
    setting.xAxisIndex = 'all';
  }
  if (setting.yAxisIndex == null && setting.yAxisId == null) {
    setting.yAxisIndex = 'all';
  }
  return setting;
}
function updateBackBtnStatus(featureModel, ecModel) {
  featureModel.setIconStatus('back', history.count(ecModel) > 1 ? 'emphasis' : 'normal');
}
function updateZoomBtnStatus(featureModel, ecModel, view, payload, api) {
  var zoomActive = view._isZoomActive;
  if (payload && payload.type === 'takeGlobalCursor') {
    zoomActive = payload.key === 'dataZoomSelect' ? payload.dataZoomSelectActive : false;
  }
  view._isZoomActive = zoomActive;
  featureModel.setIconStatus('zoom', zoomActive ? 'emphasis' : 'normal');
  var brushTargetManager = new BrushTargetManager(makeAxisFinder(featureModel), ecModel, {
    include: ['grid']
  });
  var panels = brushTargetManager.makePanelOpts(api, function (targetInfo) {
    return targetInfo.xAxisDeclared && !targetInfo.yAxisDeclared ? 'lineX' : !targetInfo.xAxisDeclared && targetInfo.yAxisDeclared ? 'lineY' : 'rect';
  });
  view._brushController.setPanels(panels).enableBrush(zoomActive && panels.length ? {
    brushType: 'auto',
    brushStyle: featureModel.getModel('brushStyle').getItemStyle()
  } : false);
}
registerInternalOptionCreator('dataZoom', function (ecModel) {
  var toolboxModel = ecModel.getComponent('toolbox', 0);
  var featureDataZoomPath = ['feature', 'dataZoom'];
  if (!toolboxModel || toolboxModel.get(featureDataZoomPath) == null) {
    return;
  }
  var dzFeatureModel = toolboxModel.getModel(featureDataZoomPath);
  var dzOptions = [];
  var finder = makeAxisFinder(dzFeatureModel);
  var finderResult = parseFinder(ecModel, finder);
  each(finderResult.xAxisModels, function (axisModel) {
    return buildInternalOptions(axisModel, 'xAxis', 'xAxisIndex');
  });
  each(finderResult.yAxisModels, function (axisModel) {
    return buildInternalOptions(axisModel, 'yAxis', 'yAxisIndex');
  });
  function buildInternalOptions(axisModel, axisMainType, axisIndexPropName) {
    var axisIndex = axisModel.componentIndex;
    var newOpt = {
      type: 'select',
      $fromToolbox: true,
      // Default to be filter
      filterMode: dzFeatureModel.get('filterMode', true) || 'filter',
      // Id for merge mapping.
      id: DATA_ZOOM_ID_BASE + axisMainType + axisIndex
    };
    newOpt[axisIndexPropName] = axisIndex;
    dzOptions.push(newOpt);
  }
  return dzOptions;
});
export default DataZoomFeature;