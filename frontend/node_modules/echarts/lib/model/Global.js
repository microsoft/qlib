
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
/**
 * Caution: If the mechanism should be changed some day, these cases
 * should be considered:
 *
 * (1) In `merge option` mode, if using the same option to call `setOption`
 * many times, the result should be the same (try our best to ensure that).
 * (2) In `merge option` mode, if a component has no id/name specified, it
 * will be merged by index, and the result sequence of the components is
 * consistent to the original sequence.
 * (3) In `replaceMerge` mode, keep the result sequence of the components is
 * consistent to the original sequence, even though there might result in "hole".
 * (4) `reset` feature (in toolbox). Find detailed info in comments about
 * `mergeOption` in module:echarts/model/OptionManager.
 */
import { each, filter, isArray, isObject, isString, createHashMap, assert, clone, merge, extend, mixin, isFunction } from 'zrender/lib/core/util.js';
import * as modelUtil from '../util/model.js';
import Model from './Model.js';
import ComponentModel from './Component.js';
import globalDefault from './globalDefault.js';
import { resetSourceDefaulter } from '../data/helper/sourceHelper.js';
import { concatInternalOptions } from './internalComponentCreator.js';
import { PaletteMixin } from './mixin/palette.js';
import { error, warn } from '../util/log.js';
// -----------------------
// Internal method names:
// -----------------------
var reCreateSeriesIndices;
var assertSeriesInitialized;
var initBase;
var OPTION_INNER_KEY = '\0_ec_inner';
var OPTION_INNER_VALUE = 1;
var BUITIN_COMPONENTS_MAP = {
  grid: 'GridComponent',
  polar: 'PolarComponent',
  geo: 'GeoComponent',
  singleAxis: 'SingleAxisComponent',
  parallel: 'ParallelComponent',
  calendar: 'CalendarComponent',
  graphic: 'GraphicComponent',
  toolbox: 'ToolboxComponent',
  tooltip: 'TooltipComponent',
  axisPointer: 'AxisPointerComponent',
  brush: 'BrushComponent',
  title: 'TitleComponent',
  timeline: 'TimelineComponent',
  markPoint: 'MarkPointComponent',
  markLine: 'MarkLineComponent',
  markArea: 'MarkAreaComponent',
  legend: 'LegendComponent',
  dataZoom: 'DataZoomComponent',
  visualMap: 'VisualMapComponent',
  // aria: 'AriaComponent',
  // dataset: 'DatasetComponent',
  // Dependencies
  xAxis: 'GridComponent',
  yAxis: 'GridComponent',
  angleAxis: 'PolarComponent',
  radiusAxis: 'PolarComponent'
};
var BUILTIN_CHARTS_MAP = {
  line: 'LineChart',
  bar: 'BarChart',
  pie: 'PieChart',
  scatter: 'ScatterChart',
  radar: 'RadarChart',
  map: 'MapChart',
  tree: 'TreeChart',
  treemap: 'TreemapChart',
  graph: 'GraphChart',
  gauge: 'GaugeChart',
  funnel: 'FunnelChart',
  parallel: 'ParallelChart',
  sankey: 'SankeyChart',
  boxplot: 'BoxplotChart',
  candlestick: 'CandlestickChart',
  effectScatter: 'EffectScatterChart',
  lines: 'LinesChart',
  heatmap: 'HeatmapChart',
  pictorialBar: 'PictorialBarChart',
  themeRiver: 'ThemeRiverChart',
  sunburst: 'SunburstChart',
  custom: 'CustomChart'
};
var componetsMissingLogPrinted = {};
function checkMissingComponents(option) {
  each(option, function (componentOption, mainType) {
    if (!ComponentModel.hasClass(mainType)) {
      var componentImportName = BUITIN_COMPONENTS_MAP[mainType];
      if (componentImportName && !componetsMissingLogPrinted[componentImportName]) {
        error("Component " + mainType + " is used but not imported.\nimport { " + componentImportName + " } from 'echarts/components';\necharts.use([" + componentImportName + "]);");
        componetsMissingLogPrinted[componentImportName] = true;
      }
    }
  });
}
var GlobalModel = /** @class */function (_super) {
  __extends(GlobalModel, _super);
  function GlobalModel() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  GlobalModel.prototype.init = function (option, parentModel, ecModel, theme, locale, optionManager) {
    theme = theme || {};
    this.option = null; // Mark as not initialized.
    this._theme = new Model(theme);
    this._locale = new Model(locale);
    this._optionManager = optionManager;
  };
  GlobalModel.prototype.setOption = function (option, opts, optionPreprocessorFuncs) {
    if (process.env.NODE_ENV !== 'production') {
      assert(option != null, 'option is null/undefined');
      assert(option[OPTION_INNER_KEY] !== OPTION_INNER_VALUE, 'please use chart.getOption()');
    }
    var innerOpt = normalizeSetOptionInput(opts);
    this._optionManager.setOption(option, optionPreprocessorFuncs, innerOpt);
    this._resetOption(null, innerOpt);
  };
  /**
   * @param type null/undefined: reset all.
   *        'recreate': force recreate all.
   *        'timeline': only reset timeline option
   *        'media': only reset media query option
   * @return Whether option changed.
   */
  GlobalModel.prototype.resetOption = function (type, opt) {
    return this._resetOption(type, normalizeSetOptionInput(opt));
  };
  GlobalModel.prototype._resetOption = function (type, opt) {
    var optionChanged = false;
    var optionManager = this._optionManager;
    if (!type || type === 'recreate') {
      var baseOption = optionManager.mountOption(type === 'recreate');
      if (process.env.NODE_ENV !== 'production') {
        checkMissingComponents(baseOption);
      }
      if (!this.option || type === 'recreate') {
        initBase(this, baseOption);
      } else {
        this.restoreData();
        this._mergeOption(baseOption, opt);
      }
      optionChanged = true;
    }
    if (type === 'timeline' || type === 'media') {
      this.restoreData();
    }
    // By design, if `setOption(option2)` at the second time, and `option2` is a `ECUnitOption`,
    // it should better not have the same props with `MediaUnit['option']`.
    // Because either `option2` or `MediaUnit['option']` will be always merged to "current option"
    // rather than original "baseOption". If they both override a prop, the result might be
    // unexpected when media state changed after `setOption` called.
    // If we really need to modify a props in each `MediaUnit['option']`, use the full version
    // (`{baseOption, media}`) in `setOption`.
    // For `timeline`, the case is the same.
    if (!type || type === 'recreate' || type === 'timeline') {
      var timelineOption = optionManager.getTimelineOption(this);
      if (timelineOption) {
        optionChanged = true;
        this._mergeOption(timelineOption, opt);
      }
    }
    if (!type || type === 'recreate' || type === 'media') {
      var mediaOptions = optionManager.getMediaOption(this);
      if (mediaOptions.length) {
        each(mediaOptions, function (mediaOption) {
          optionChanged = true;
          this._mergeOption(mediaOption, opt);
        }, this);
      }
    }
    return optionChanged;
  };
  GlobalModel.prototype.mergeOption = function (option) {
    this._mergeOption(option, null);
  };
  GlobalModel.prototype._mergeOption = function (newOption, opt) {
    var option = this.option;
    var componentsMap = this._componentsMap;
    var componentsCount = this._componentsCount;
    var newCmptTypes = [];
    var newCmptTypeMap = createHashMap();
    var replaceMergeMainTypeMap = opt && opt.replaceMergeMainTypeMap;
    resetSourceDefaulter(this);
    // If no component class, merge directly.
    // For example: color, animaiton options, etc.
    each(newOption, function (componentOption, mainType) {
      if (componentOption == null) {
        return;
      }
      if (!ComponentModel.hasClass(mainType)) {
        // globalSettingTask.dirty();
        option[mainType] = option[mainType] == null ? clone(componentOption) : merge(option[mainType], componentOption, true);
      } else if (mainType) {
        newCmptTypes.push(mainType);
        newCmptTypeMap.set(mainType, true);
      }
    });
    if (replaceMergeMainTypeMap) {
      // If there is a mainType `xxx` in `replaceMerge` but not declared in option,
      // we trade it as it is declared in option as `{xxx: []}`. Because:
      // (1) for normal merge, `{xxx: null/undefined}` are the same meaning as `{xxx: []}`.
      // (2) some preprocessor may convert some of `{xxx: null/undefined}` to `{xxx: []}`.
      replaceMergeMainTypeMap.each(function (val, mainTypeInReplaceMerge) {
        if (ComponentModel.hasClass(mainTypeInReplaceMerge) && !newCmptTypeMap.get(mainTypeInReplaceMerge)) {
          newCmptTypes.push(mainTypeInReplaceMerge);
          newCmptTypeMap.set(mainTypeInReplaceMerge, true);
        }
      });
    }
    ComponentModel.topologicalTravel(newCmptTypes, ComponentModel.getAllClassMainTypes(), visitComponent, this);
    function visitComponent(mainType) {
      var newCmptOptionList = concatInternalOptions(this, mainType, modelUtil.normalizeToArray(newOption[mainType]));
      var oldCmptList = componentsMap.get(mainType);
      var mergeMode =
      // `!oldCmptList` means init. See the comment in `mappingToExists`
      !oldCmptList ? 'replaceAll' : replaceMergeMainTypeMap && replaceMergeMainTypeMap.get(mainType) ? 'replaceMerge' : 'normalMerge';
      var mappingResult = modelUtil.mappingToExists(oldCmptList, newCmptOptionList, mergeMode);
      // Set mainType and complete subType.
      modelUtil.setComponentTypeToKeyInfo(mappingResult, mainType, ComponentModel);
      // Empty it before the travel, in order to prevent `this._componentsMap`
      // from being used in the `init`/`mergeOption`/`optionUpdated` of some
      // components, which is probably incorrect logic.
      option[mainType] = null;
      componentsMap.set(mainType, null);
      componentsCount.set(mainType, 0);
      var optionsByMainType = [];
      var cmptsByMainType = [];
      var cmptsCountByMainType = 0;
      var tooltipExists;
      var tooltipWarningLogged;
      each(mappingResult, function (resultItem, index) {
        var componentModel = resultItem.existing;
        var newCmptOption = resultItem.newOption;
        if (!newCmptOption) {
          if (componentModel) {
            // Consider where is no new option and should be merged using {},
            // see removeEdgeAndAdd in topologicalTravel and
            // ComponentModel.getAllClassMainTypes.
            componentModel.mergeOption({}, this);
            componentModel.optionUpdated({}, false);
          }
          // If no both `resultItem.exist` and `resultItem.option`,
          // either it is in `replaceMerge` and not matched by any id,
          // or it has been removed in previous `replaceMerge` and left a "hole" in this component index.
        } else {
          var isSeriesType = mainType === 'series';
          var ComponentModelClass = ComponentModel.getClass(mainType, resultItem.keyInfo.subType, !isSeriesType // Give a more detailed warn later if series don't exists
          );
          if (!ComponentModelClass) {
            if (process.env.NODE_ENV !== 'production') {
              var subType = resultItem.keyInfo.subType;
              var seriesImportName = BUILTIN_CHARTS_MAP[subType];
              if (!componetsMissingLogPrinted[subType]) {
                componetsMissingLogPrinted[subType] = true;
                if (seriesImportName) {
                  error("Series " + subType + " is used but not imported.\nimport { " + seriesImportName + " } from 'echarts/charts';\necharts.use([" + seriesImportName + "]);");
                } else {
                  error("Unknown series " + subType);
                }
              }
            }
            return;
          }
          // TODO Before multiple tooltips get supported, we do this check to avoid unexpected exception.
          if (mainType === 'tooltip') {
            if (tooltipExists) {
              if (process.env.NODE_ENV !== 'production') {
                if (!tooltipWarningLogged) {
                  warn('Currently only one tooltip component is allowed.');
                  tooltipWarningLogged = true;
                }
              }
              return;
            }
            tooltipExists = true;
          }
          if (componentModel && componentModel.constructor === ComponentModelClass) {
            componentModel.name = resultItem.keyInfo.name;
            // componentModel.settingTask && componentModel.settingTask.dirty();
            componentModel.mergeOption(newCmptOption, this);
            componentModel.optionUpdated(newCmptOption, false);
          } else {
            // PENDING Global as parent ?
            var extraOpt = extend({
              componentIndex: index
            }, resultItem.keyInfo);
            componentModel = new ComponentModelClass(newCmptOption, this, this, extraOpt);
            // Assign `keyInfo`
            extend(componentModel, extraOpt);
            if (resultItem.brandNew) {
              componentModel.__requireNewView = true;
            }
            componentModel.init(newCmptOption, this, this);
            // Call optionUpdated after init.
            // newCmptOption has been used as componentModel.option
            // and may be merged with theme and default, so pass null
            // to avoid confusion.
            componentModel.optionUpdated(null, true);
          }
        }
        if (componentModel) {
          optionsByMainType.push(componentModel.option);
          cmptsByMainType.push(componentModel);
          cmptsCountByMainType++;
        } else {
          // Always do assign to avoid elided item in array.
          optionsByMainType.push(void 0);
          cmptsByMainType.push(void 0);
        }
      }, this);
      option[mainType] = optionsByMainType;
      componentsMap.set(mainType, cmptsByMainType);
      componentsCount.set(mainType, cmptsCountByMainType);
      // Backup series for filtering.
      if (mainType === 'series') {
        reCreateSeriesIndices(this);
      }
    }
    // If no series declared, ensure `_seriesIndices` initialized.
    if (!this._seriesIndices) {
      reCreateSeriesIndices(this);
    }
  };
  /**
   * Get option for output (cloned option and inner info removed)
   */
  GlobalModel.prototype.getOption = function () {
    var option = clone(this.option);
    each(option, function (optInMainType, mainType) {
      if (ComponentModel.hasClass(mainType)) {
        var opts = modelUtil.normalizeToArray(optInMainType);
        // Inner cmpts need to be removed.
        // Inner cmpts might not be at last since ec5.0, but still
        // compatible for users: if inner cmpt at last, splice the returned array.
        var realLen = opts.length;
        var metNonInner = false;
        for (var i = realLen - 1; i >= 0; i--) {
          // Remove options with inner id.
          if (opts[i] && !modelUtil.isComponentIdInternal(opts[i])) {
            metNonInner = true;
          } else {
            opts[i] = null;
            !metNonInner && realLen--;
          }
        }
        opts.length = realLen;
        option[mainType] = opts;
      }
    });
    delete option[OPTION_INNER_KEY];
    return option;
  };
  GlobalModel.prototype.getTheme = function () {
    return this._theme;
  };
  GlobalModel.prototype.getLocaleModel = function () {
    return this._locale;
  };
  GlobalModel.prototype.setUpdatePayload = function (payload) {
    this._payload = payload;
  };
  GlobalModel.prototype.getUpdatePayload = function () {
    return this._payload;
  };
  /**
   * @param idx If not specified, return the first one.
   */
  GlobalModel.prototype.getComponent = function (mainType, idx) {
    var list = this._componentsMap.get(mainType);
    if (list) {
      var cmpt = list[idx || 0];
      if (cmpt) {
        return cmpt;
      } else if (idx == null) {
        for (var i = 0; i < list.length; i++) {
          if (list[i]) {
            return list[i];
          }
        }
      }
    }
  };
  /**
   * @return Never be null/undefined.
   */
  GlobalModel.prototype.queryComponents = function (condition) {
    var mainType = condition.mainType;
    if (!mainType) {
      return [];
    }
    var index = condition.index;
    var id = condition.id;
    var name = condition.name;
    var cmpts = this._componentsMap.get(mainType);
    if (!cmpts || !cmpts.length) {
      return [];
    }
    var result;
    if (index != null) {
      result = [];
      each(modelUtil.normalizeToArray(index), function (idx) {
        cmpts[idx] && result.push(cmpts[idx]);
      });
    } else if (id != null) {
      result = queryByIdOrName('id', id, cmpts);
    } else if (name != null) {
      result = queryByIdOrName('name', name, cmpts);
    } else {
      // Return all non-empty components in that mainType
      result = filter(cmpts, function (cmpt) {
        return !!cmpt;
      });
    }
    return filterBySubType(result, condition);
  };
  /**
   * The interface is different from queryComponents,
   * which is convenient for inner usage.
   *
   * @usage
   * let result = findComponents(
   *     {mainType: 'dataZoom', query: {dataZoomId: 'abc'}}
   * );
   * let result = findComponents(
   *     {mainType: 'series', subType: 'pie', query: {seriesName: 'uio'}}
   * );
   * let result = findComponents(
   *     {mainType: 'series',
   *     filter: function (model, index) {...}}
   * );
   * // result like [component0, componnet1, ...]
   */
  GlobalModel.prototype.findComponents = function (condition) {
    var query = condition.query;
    var mainType = condition.mainType;
    var queryCond = getQueryCond(query);
    var result = queryCond ? this.queryComponents(queryCond)
    // Retrieve all non-empty components.
    : filter(this._componentsMap.get(mainType), function (cmpt) {
      return !!cmpt;
    });
    return doFilter(filterBySubType(result, condition));
    function getQueryCond(q) {
      var indexAttr = mainType + 'Index';
      var idAttr = mainType + 'Id';
      var nameAttr = mainType + 'Name';
      return q && (q[indexAttr] != null || q[idAttr] != null || q[nameAttr] != null) ? {
        mainType: mainType,
        // subType will be filtered finally.
        index: q[indexAttr],
        id: q[idAttr],
        name: q[nameAttr]
      } : null;
    }
    function doFilter(res) {
      return condition.filter ? filter(res, condition.filter) : res;
    }
  };
  GlobalModel.prototype.eachComponent = function (mainType, cb, context) {
    var componentsMap = this._componentsMap;
    if (isFunction(mainType)) {
      var ctxForAll_1 = cb;
      var cbForAll_1 = mainType;
      componentsMap.each(function (cmpts, componentType) {
        for (var i = 0; cmpts && i < cmpts.length; i++) {
          var cmpt = cmpts[i];
          cmpt && cbForAll_1.call(ctxForAll_1, componentType, cmpt, cmpt.componentIndex);
        }
      });
    } else {
      var cmpts = isString(mainType) ? componentsMap.get(mainType) : isObject(mainType) ? this.findComponents(mainType) : null;
      for (var i = 0; cmpts && i < cmpts.length; i++) {
        var cmpt = cmpts[i];
        cmpt && cb.call(context, cmpt, cmpt.componentIndex);
      }
    }
  };
  /**
   * Get series list before filtered by name.
   */
  GlobalModel.prototype.getSeriesByName = function (name) {
    var nameStr = modelUtil.convertOptionIdName(name, null);
    return filter(this._componentsMap.get('series'), function (oneSeries) {
      return !!oneSeries && nameStr != null && oneSeries.name === nameStr;
    });
  };
  /**
   * Get series list before filtered by index.
   */
  GlobalModel.prototype.getSeriesByIndex = function (seriesIndex) {
    return this._componentsMap.get('series')[seriesIndex];
  };
  /**
   * Get series list before filtered by type.
   * FIXME: rename to getRawSeriesByType?
   */
  GlobalModel.prototype.getSeriesByType = function (subType) {
    return filter(this._componentsMap.get('series'), function (oneSeries) {
      return !!oneSeries && oneSeries.subType === subType;
    });
  };
  /**
   * Get all series before filtered.
   */
  GlobalModel.prototype.getSeries = function () {
    return filter(this._componentsMap.get('series'), function (oneSeries) {
      return !!oneSeries;
    });
  };
  /**
   * Count series before filtered.
   */
  GlobalModel.prototype.getSeriesCount = function () {
    return this._componentsCount.get('series');
  };
  /**
   * After filtering, series may be different
   * from raw series.
   */
  GlobalModel.prototype.eachSeries = function (cb, context) {
    assertSeriesInitialized(this);
    each(this._seriesIndices, function (rawSeriesIndex) {
      var series = this._componentsMap.get('series')[rawSeriesIndex];
      cb.call(context, series, rawSeriesIndex);
    }, this);
  };
  /**
   * Iterate raw series before filtered.
   *
   * @param {Function} cb
   * @param {*} context
   */
  GlobalModel.prototype.eachRawSeries = function (cb, context) {
    each(this._componentsMap.get('series'), function (series) {
      series && cb.call(context, series, series.componentIndex);
    });
  };
  /**
   * After filtering, series may be different.
   * from raw series.
   */
  GlobalModel.prototype.eachSeriesByType = function (subType, cb, context) {
    assertSeriesInitialized(this);
    each(this._seriesIndices, function (rawSeriesIndex) {
      var series = this._componentsMap.get('series')[rawSeriesIndex];
      if (series.subType === subType) {
        cb.call(context, series, rawSeriesIndex);
      }
    }, this);
  };
  /**
   * Iterate raw series before filtered of given type.
   */
  GlobalModel.prototype.eachRawSeriesByType = function (subType, cb, context) {
    return each(this.getSeriesByType(subType), cb, context);
  };
  GlobalModel.prototype.isSeriesFiltered = function (seriesModel) {
    assertSeriesInitialized(this);
    return this._seriesIndicesMap.get(seriesModel.componentIndex) == null;
  };
  GlobalModel.prototype.getCurrentSeriesIndices = function () {
    return (this._seriesIndices || []).slice();
  };
  GlobalModel.prototype.filterSeries = function (cb, context) {
    assertSeriesInitialized(this);
    var newSeriesIndices = [];
    each(this._seriesIndices, function (seriesRawIdx) {
      var series = this._componentsMap.get('series')[seriesRawIdx];
      cb.call(context, series, seriesRawIdx) && newSeriesIndices.push(seriesRawIdx);
    }, this);
    this._seriesIndices = newSeriesIndices;
    this._seriesIndicesMap = createHashMap(newSeriesIndices);
  };
  GlobalModel.prototype.restoreData = function (payload) {
    reCreateSeriesIndices(this);
    var componentsMap = this._componentsMap;
    var componentTypes = [];
    componentsMap.each(function (components, componentType) {
      if (ComponentModel.hasClass(componentType)) {
        componentTypes.push(componentType);
      }
    });
    ComponentModel.topologicalTravel(componentTypes, ComponentModel.getAllClassMainTypes(), function (componentType) {
      each(componentsMap.get(componentType), function (component) {
        if (component && (componentType !== 'series' || !isNotTargetSeries(component, payload))) {
          component.restoreData();
        }
      });
    });
  };
  GlobalModel.internalField = function () {
    reCreateSeriesIndices = function (ecModel) {
      var seriesIndices = ecModel._seriesIndices = [];
      each(ecModel._componentsMap.get('series'), function (series) {
        // series may have been removed by `replaceMerge`.
        series && seriesIndices.push(series.componentIndex);
      });
      ecModel._seriesIndicesMap = createHashMap(seriesIndices);
    };
    assertSeriesInitialized = function (ecModel) {
      // Components that use _seriesIndices should depends on series component,
      // which make sure that their initialization is after series.
      if (process.env.NODE_ENV !== 'production') {
        if (!ecModel._seriesIndices) {
          throw new Error('Option should contains series.');
        }
      }
    };
    initBase = function (ecModel, baseOption) {
      // Using OPTION_INNER_KEY to mark that this option cannot be used outside,
      // i.e. `chart.setOption(chart.getModel().option);` is forbidden.
      ecModel.option = {};
      ecModel.option[OPTION_INNER_KEY] = OPTION_INNER_VALUE;
      // Init with series: [], in case of calling findSeries method
      // before series initialized.
      ecModel._componentsMap = createHashMap({
        series: []
      });
      ecModel._componentsCount = createHashMap();
      // If user spefied `option.aria`, aria will be enable. This detection should be
      // performed before theme and globalDefault merge.
      var airaOption = baseOption.aria;
      if (isObject(airaOption) && airaOption.enabled == null) {
        airaOption.enabled = true;
      }
      mergeTheme(baseOption, ecModel._theme.option);
      // TODO Needs clone when merging to the unexisted property
      merge(baseOption, globalDefault, false);
      ecModel._mergeOption(baseOption, null);
    };
  }();
  return GlobalModel;
}(Model);
function isNotTargetSeries(seriesModel, payload) {
  if (payload) {
    var index = payload.seriesIndex;
    var id = payload.seriesId;
    var name_1 = payload.seriesName;
    return index != null && seriesModel.componentIndex !== index || id != null && seriesModel.id !== id || name_1 != null && seriesModel.name !== name_1;
  }
}
function mergeTheme(option, theme) {
  // PENDING
  // NOT use `colorLayer` in theme if option has `color`
  var notMergeColorLayer = option.color && !option.colorLayer;
  each(theme, function (themeItem, name) {
    if (name === 'colorLayer' && notMergeColorLayer) {
      return;
    }
    // If it is component model mainType, the model handles that merge later.
    // otherwise, merge them here.
    if (!ComponentModel.hasClass(name)) {
      if (typeof themeItem === 'object') {
        option[name] = !option[name] ? clone(themeItem) : merge(option[name], themeItem, false);
      } else {
        if (option[name] == null) {
          option[name] = themeItem;
        }
      }
    }
  });
}
function queryByIdOrName(attr, idOrName, cmpts) {
  // Here is a break from echarts4: string and number are
  // treated as equal.
  if (isArray(idOrName)) {
    var keyMap_1 = createHashMap();
    each(idOrName, function (idOrNameItem) {
      if (idOrNameItem != null) {
        var idName = modelUtil.convertOptionIdName(idOrNameItem, null);
        idName != null && keyMap_1.set(idOrNameItem, true);
      }
    });
    return filter(cmpts, function (cmpt) {
      return cmpt && keyMap_1.get(cmpt[attr]);
    });
  } else {
    var idName_1 = modelUtil.convertOptionIdName(idOrName, null);
    return filter(cmpts, function (cmpt) {
      return cmpt && idName_1 != null && cmpt[attr] === idName_1;
    });
  }
}
function filterBySubType(components, condition) {
  // Using hasOwnProperty for restrict. Consider
  // subType is undefined in user payload.
  return condition.hasOwnProperty('subType') ? filter(components, function (cmpt) {
    return cmpt && cmpt.subType === condition.subType;
  }) : components;
}
function normalizeSetOptionInput(opts) {
  var replaceMergeMainTypeMap = createHashMap();
  opts && each(modelUtil.normalizeToArray(opts.replaceMerge), function (mainType) {
    if (process.env.NODE_ENV !== 'production') {
      assert(ComponentModel.hasClass(mainType), '"' + mainType + '" is not valid component main type in "replaceMerge"');
    }
    replaceMergeMainTypeMap.set(mainType, true);
  });
  return {
    replaceMergeMainTypeMap: replaceMergeMainTypeMap
  };
}
mixin(GlobalModel, PaletteMixin);
export default GlobalModel;