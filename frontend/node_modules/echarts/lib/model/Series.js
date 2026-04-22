
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
import * as modelUtil from '../util/model.js';
import ComponentModel from './Component.js';
import { PaletteMixin } from './mixin/palette.js';
import { DataFormatMixin } from '../model/mixin/dataFormat.js';
import { getLayoutParams, mergeLayoutParam, fetchLayoutMode } from '../util/layout.js';
import { createTask } from '../core/task.js';
import { mountExtend } from '../util/clazz.js';
import { SourceManager } from '../data/helper/sourceManager.js';
import { defaultSeriesFormatTooltip } from '../component/tooltip/seriesFormatTooltip.js';
var inner = modelUtil.makeInner();
function getSelectionKey(data, dataIndex) {
  return data.getName(dataIndex) || data.getId(dataIndex);
}
export var SERIES_UNIVERSAL_TRANSITION_PROP = '__universalTransitionEnabled';
var SeriesModel = /** @class */function (_super) {
  __extends(SeriesModel, _super);
  function SeriesModel() {
    // [Caution]: Because this class or desecendants can be used as `XXX.extend(subProto)`,
    // the class members must not be initialized in constructor or declaration place.
    // Otherwise there is bad case:
    //   class A {xxx = 1;}
    //   enableClassExtend(A);
    //   class B extends A {}
    //   var C = B.extend({xxx: 5});
    //   var c = new C();
    //   console.log(c.xxx); // expect 5 but always 1.
    var _this = _super !== null && _super.apply(this, arguments) || this;
    // ---------------------------------------
    // Props about data selection
    // ---------------------------------------
    _this._selectedDataIndicesMap = {};
    return _this;
  }
  SeriesModel.prototype.init = function (option, parentModel, ecModel) {
    this.seriesIndex = this.componentIndex;
    this.dataTask = createTask({
      count: dataTaskCount,
      reset: dataTaskReset
    });
    this.dataTask.context = {
      model: this
    };
    this.mergeDefaultAndTheme(option, ecModel);
    var sourceManager = inner(this).sourceManager = new SourceManager(this);
    sourceManager.prepareSource();
    var data = this.getInitialData(option, ecModel);
    wrapData(data, this);
    this.dataTask.context.data = data;
    if (process.env.NODE_ENV !== 'production') {
      zrUtil.assert(data, 'getInitialData returned invalid data.');
    }
    inner(this).dataBeforeProcessed = data;
    // If we reverse the order (make data firstly, and then make
    // dataBeforeProcessed by cloneShallow), cloneShallow will
    // cause data.graph.data !== data when using
    // module:echarts/data/Graph or module:echarts/data/Tree.
    // See module:echarts/data/helper/linkSeriesData
    // Theoretically, it is unreasonable to call `seriesModel.getData()` in the model
    // init or merge stage, because the data can be restored. So we do not `restoreData`
    // and `setData` here, which forbids calling `seriesModel.getData()` in this stage.
    // Call `seriesModel.getRawData()` instead.
    // this.restoreData();
    autoSeriesName(this);
    this._initSelectedMapFromData(data);
  };
  /**
   * Util for merge default and theme to option
   */
  SeriesModel.prototype.mergeDefaultAndTheme = function (option, ecModel) {
    var layoutMode = fetchLayoutMode(this);
    var inputPositionParams = layoutMode ? getLayoutParams(option) : {};
    // Backward compat: using subType on theme.
    // But if name duplicate between series subType
    // (for example: parallel) add component mainType,
    // add suffix 'Series'.
    var themeSubType = this.subType;
    if (ComponentModel.hasClass(themeSubType)) {
      themeSubType += 'Series';
    }
    zrUtil.merge(option, ecModel.getTheme().get(this.subType));
    zrUtil.merge(option, this.getDefaultOption());
    // Default label emphasis `show`
    modelUtil.defaultEmphasis(option, 'label', ['show']);
    this.fillDataTextStyle(option.data);
    if (layoutMode) {
      mergeLayoutParam(option, inputPositionParams, layoutMode);
    }
  };
  SeriesModel.prototype.mergeOption = function (newSeriesOption, ecModel) {
    // this.settingTask.dirty();
    newSeriesOption = zrUtil.merge(this.option, newSeriesOption, true);
    this.fillDataTextStyle(newSeriesOption.data);
    var layoutMode = fetchLayoutMode(this);
    if (layoutMode) {
      mergeLayoutParam(this.option, newSeriesOption, layoutMode);
    }
    var sourceManager = inner(this).sourceManager;
    sourceManager.dirty();
    sourceManager.prepareSource();
    var data = this.getInitialData(newSeriesOption, ecModel);
    wrapData(data, this);
    this.dataTask.dirty();
    this.dataTask.context.data = data;
    inner(this).dataBeforeProcessed = data;
    autoSeriesName(this);
    this._initSelectedMapFromData(data);
  };
  SeriesModel.prototype.fillDataTextStyle = function (data) {
    // Default data label emphasis `show`
    // FIXME Tree structure data ?
    // FIXME Performance ?
    if (data && !zrUtil.isTypedArray(data)) {
      var props = ['show'];
      for (var i = 0; i < data.length; i++) {
        if (data[i] && data[i].label) {
          modelUtil.defaultEmphasis(data[i], 'label', props);
        }
      }
    }
  };
  /**
   * Init a data structure from data related option in series
   * Must be overridden.
   */
  SeriesModel.prototype.getInitialData = function (option, ecModel) {
    return;
  };
  /**
   * Append data to list
   */
  SeriesModel.prototype.appendData = function (params) {
    // FIXME ???
    // (1) If data from dataset, forbidden append.
    // (2) support append data of dataset.
    var data = this.getRawData();
    data.appendData(params.data);
  };
  /**
   * Consider some method like `filter`, `map` need make new data,
   * We should make sure that `seriesModel.getData()` get correct
   * data in the stream procedure. So we fetch data from upstream
   * each time `task.perform` called.
   */
  SeriesModel.prototype.getData = function (dataType) {
    var task = getCurrentTask(this);
    if (task) {
      var data = task.context.data;
      return dataType == null || !data.getLinkedData ? data : data.getLinkedData(dataType);
    } else {
      // When series is not alive (that may happen when click toolbox
      // restore or setOption with not merge mode), series data may
      // be still need to judge animation or something when graphic
      // elements want to know whether fade out.
      return inner(this).data;
    }
  };
  SeriesModel.prototype.getAllData = function () {
    var mainData = this.getData();
    return mainData && mainData.getLinkedDataAll ? mainData.getLinkedDataAll() : [{
      data: mainData
    }];
  };
  SeriesModel.prototype.setData = function (data) {
    var task = getCurrentTask(this);
    if (task) {
      var context = task.context;
      // Consider case: filter, data sample.
      // FIXME:TS never used, so comment it
      // if (context.data !== data && task.modifyOutputEnd) {
      //     task.setOutputEnd(data.count());
      // }
      context.outputData = data;
      // Caution: setData should update context.data,
      // Because getData may be called multiply in a
      // single stage and expect to get the data just
      // set. (For example, AxisProxy, x y both call
      // getData and setDate sequentially).
      // So the context.data should be fetched from
      // upstream each time when a stage starts to be
      // performed.
      if (task !== this.dataTask) {
        context.data = data;
      }
    }
    inner(this).data = data;
  };
  SeriesModel.prototype.getEncode = function () {
    var encode = this.get('encode', true);
    if (encode) {
      return zrUtil.createHashMap(encode);
    }
  };
  SeriesModel.prototype.getSourceManager = function () {
    return inner(this).sourceManager;
  };
  SeriesModel.prototype.getSource = function () {
    return this.getSourceManager().getSource();
  };
  /**
   * Get data before processed
   */
  SeriesModel.prototype.getRawData = function () {
    return inner(this).dataBeforeProcessed;
  };
  SeriesModel.prototype.getColorBy = function () {
    var colorBy = this.get('colorBy');
    return colorBy || 'series';
  };
  SeriesModel.prototype.isColorBySeries = function () {
    return this.getColorBy() === 'series';
  };
  /**
   * Get base axis if has coordinate system and has axis.
   * By default use coordSys.getBaseAxis();
   * Can be overridden for some chart.
   * @return {type} description
   */
  SeriesModel.prototype.getBaseAxis = function () {
    var coordSys = this.coordinateSystem;
    // @ts-ignore
    return coordSys && coordSys.getBaseAxis && coordSys.getBaseAxis();
  };
  /**
   * Default tooltip formatter
   *
   * @param dataIndex
   * @param multipleSeries
   * @param dataType
   * @param renderMode valid values: 'html'(by default) and 'richText'.
   *        'html' is used for rendering tooltip in extra DOM form, and the result
   *        string is used as DOM HTML content.
   *        'richText' is used for rendering tooltip in rich text form, for those where
   *        DOM operation is not supported.
   * @return formatted tooltip with `html` and `markers`
   *        Notice: The override method can also return string
   */
  SeriesModel.prototype.formatTooltip = function (dataIndex, multipleSeries, dataType) {
    return defaultSeriesFormatTooltip({
      series: this,
      dataIndex: dataIndex,
      multipleSeries: multipleSeries
    });
  };
  SeriesModel.prototype.isAnimationEnabled = function () {
    var ecModel = this.ecModel;
    // Disable animation if using echarts in node but not give ssr flag.
    // In ssr mode, renderToString will generate svg with css animation.
    if (env.node && !(ecModel && ecModel.ssr)) {
      return false;
    }
    var animationEnabled = this.getShallow('animation');
    if (animationEnabled) {
      if (this.getData().count() > this.getShallow('animationThreshold')) {
        animationEnabled = false;
      }
    }
    return !!animationEnabled;
  };
  SeriesModel.prototype.restoreData = function () {
    this.dataTask.dirty();
  };
  SeriesModel.prototype.getColorFromPalette = function (name, scope, requestColorNum) {
    var ecModel = this.ecModel;
    // PENDING
    var color = PaletteMixin.prototype.getColorFromPalette.call(this, name, scope, requestColorNum);
    if (!color) {
      color = ecModel.getColorFromPalette(name, scope, requestColorNum);
    }
    return color;
  };
  /**
   * Use `data.mapDimensionsAll(coordDim)` instead.
   * @deprecated
   */
  SeriesModel.prototype.coordDimToDataDim = function (coordDim) {
    return this.getRawData().mapDimensionsAll(coordDim);
  };
  /**
   * Get progressive rendering count each step
   */
  SeriesModel.prototype.getProgressive = function () {
    return this.get('progressive');
  };
  /**
   * Get progressive rendering count each step
   */
  SeriesModel.prototype.getProgressiveThreshold = function () {
    return this.get('progressiveThreshold');
  };
  // PENGING If selectedMode is null ?
  SeriesModel.prototype.select = function (innerDataIndices, dataType) {
    this._innerSelect(this.getData(dataType), innerDataIndices);
  };
  SeriesModel.prototype.unselect = function (innerDataIndices, dataType) {
    var selectedMap = this.option.selectedMap;
    if (!selectedMap) {
      return;
    }
    var selectedMode = this.option.selectedMode;
    var data = this.getData(dataType);
    if (selectedMode === 'series' || selectedMap === 'all') {
      this.option.selectedMap = {};
      this._selectedDataIndicesMap = {};
      return;
    }
    for (var i = 0; i < innerDataIndices.length; i++) {
      var dataIndex = innerDataIndices[i];
      var nameOrId = getSelectionKey(data, dataIndex);
      selectedMap[nameOrId] = false;
      this._selectedDataIndicesMap[nameOrId] = -1;
    }
  };
  SeriesModel.prototype.toggleSelect = function (innerDataIndices, dataType) {
    var tmpArr = [];
    for (var i = 0; i < innerDataIndices.length; i++) {
      tmpArr[0] = innerDataIndices[i];
      this.isSelected(innerDataIndices[i], dataType) ? this.unselect(tmpArr, dataType) : this.select(tmpArr, dataType);
    }
  };
  SeriesModel.prototype.getSelectedDataIndices = function () {
    if (this.option.selectedMap === 'all') {
      return [].slice.call(this.getData().getIndices());
    }
    var selectedDataIndicesMap = this._selectedDataIndicesMap;
    var nameOrIds = zrUtil.keys(selectedDataIndicesMap);
    var dataIndices = [];
    for (var i = 0; i < nameOrIds.length; i++) {
      var dataIndex = selectedDataIndicesMap[nameOrIds[i]];
      if (dataIndex >= 0) {
        dataIndices.push(dataIndex);
      }
    }
    return dataIndices;
  };
  SeriesModel.prototype.isSelected = function (dataIndex, dataType) {
    var selectedMap = this.option.selectedMap;
    if (!selectedMap) {
      return false;
    }
    var data = this.getData(dataType);
    return (selectedMap === 'all' || selectedMap[getSelectionKey(data, dataIndex)]) && !data.getItemModel(dataIndex).get(['select', 'disabled']);
  };
  SeriesModel.prototype.isUniversalTransitionEnabled = function () {
    if (this[SERIES_UNIVERSAL_TRANSITION_PROP]) {
      return true;
    }
    var universalTransitionOpt = this.option.universalTransition;
    // Quick reject
    if (!universalTransitionOpt) {
      return false;
    }
    if (universalTransitionOpt === true) {
      return true;
    }
    // Can be simply 'universalTransition: true'
    return universalTransitionOpt && universalTransitionOpt.enabled;
  };
  SeriesModel.prototype._innerSelect = function (data, innerDataIndices) {
    var _a, _b;
    var option = this.option;
    var selectedMode = option.selectedMode;
    var len = innerDataIndices.length;
    if (!selectedMode || !len) {
      return;
    }
    if (selectedMode === 'series') {
      option.selectedMap = 'all';
    } else if (selectedMode === 'multiple') {
      if (!zrUtil.isObject(option.selectedMap)) {
        option.selectedMap = {};
      }
      var selectedMap = option.selectedMap;
      for (var i = 0; i < len; i++) {
        var dataIndex = innerDataIndices[i];
        // TODO different types of data share same object.
        var nameOrId = getSelectionKey(data, dataIndex);
        selectedMap[nameOrId] = true;
        this._selectedDataIndicesMap[nameOrId] = data.getRawIndex(dataIndex);
      }
    } else if (selectedMode === 'single' || selectedMode === true) {
      var lastDataIndex = innerDataIndices[len - 1];
      var nameOrId = getSelectionKey(data, lastDataIndex);
      option.selectedMap = (_a = {}, _a[nameOrId] = true, _a);
      this._selectedDataIndicesMap = (_b = {}, _b[nameOrId] = data.getRawIndex(lastDataIndex), _b);
    }
  };
  SeriesModel.prototype._initSelectedMapFromData = function (data) {
    // Ignore select info in data if selectedMap exists.
    // NOTE It's only for legacy usage. edge data is not supported.
    if (this.option.selectedMap) {
      return;
    }
    var dataIndices = [];
    if (data.hasItemOption) {
      data.each(function (idx) {
        var rawItem = data.getRawDataItem(idx);
        if (rawItem && rawItem.selected) {
          dataIndices.push(idx);
        }
      });
    }
    if (dataIndices.length > 0) {
      this._innerSelect(data, dataIndices);
    }
  };
  // /**
  //  * @see {module:echarts/stream/Scheduler}
  //  */
  // abstract pipeTask: null
  SeriesModel.registerClass = function (clz) {
    return ComponentModel.registerClass(clz);
  };
  SeriesModel.protoInitialize = function () {
    var proto = SeriesModel.prototype;
    proto.type = 'series.__base__';
    proto.seriesIndex = 0;
    proto.ignoreStyleOnData = false;
    proto.hasSymbolVisual = false;
    proto.defaultSymbol = 'circle';
    // Make sure the values can be accessed!
    proto.visualStyleAccessPath = 'itemStyle';
    proto.visualDrawType = 'fill';
  }();
  return SeriesModel;
}(ComponentModel);
zrUtil.mixin(SeriesModel, DataFormatMixin);
zrUtil.mixin(SeriesModel, PaletteMixin);
mountExtend(SeriesModel, ComponentModel);
/**
 * MUST be called after `prepareSource` called
 * Here we need to make auto series, especially for auto legend. But we
 * do not modify series.name in option to avoid side effects.
 */
function autoSeriesName(seriesModel) {
  // User specified name has higher priority, otherwise it may cause
  // series can not be queried unexpectedly.
  var name = seriesModel.name;
  if (!modelUtil.isNameSpecified(seriesModel)) {
    seriesModel.name = getSeriesAutoName(seriesModel) || name;
  }
}
function getSeriesAutoName(seriesModel) {
  var data = seriesModel.getRawData();
  var dataDims = data.mapDimensionsAll('seriesName');
  var nameArr = [];
  zrUtil.each(dataDims, function (dataDim) {
    var dimInfo = data.getDimensionInfo(dataDim);
    dimInfo.displayName && nameArr.push(dimInfo.displayName);
  });
  return nameArr.join(' ');
}
function dataTaskCount(context) {
  return context.model.getRawData().count();
}
function dataTaskReset(context) {
  var seriesModel = context.model;
  seriesModel.setData(seriesModel.getRawData().cloneShallow());
  return dataTaskProgress;
}
function dataTaskProgress(param, context) {
  // Avoid repeat cloneShallow when data just created in reset.
  if (context.outputData && param.end > context.outputData.count()) {
    context.model.getRawData().cloneShallow(context.outputData);
  }
}
// TODO refactor
function wrapData(data, seriesModel) {
  zrUtil.each(zrUtil.concatArray(data.CHANGABLE_METHODS, data.DOWNSAMPLE_METHODS), function (methodName) {
    data.wrapMethod(methodName, zrUtil.curry(onDataChange, seriesModel));
  });
}
function onDataChange(seriesModel, newList) {
  var task = getCurrentTask(seriesModel);
  if (task) {
    // Consider case: filter, selectRange
    task.setOutputEnd((newList || this).count());
  }
  return newList;
}
function getCurrentTask(seriesModel) {
  var scheduler = (seriesModel.ecModel || {}).scheduler;
  var pipeline = scheduler && scheduler.getPipeline(seriesModel.uid);
  if (pipeline) {
    // When pipline finished, the currrentTask keep the last
    // task (renderTask).
    var task = pipeline.currentTask;
    if (task) {
      var agentStubMap = task.agentStubMap;
      if (agentStubMap) {
        task = agentStubMap.get(seriesModel.uid);
      }
    }
    return task;
  }
}
export default SeriesModel;