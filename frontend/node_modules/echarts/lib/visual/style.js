
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
import { isFunction, extend, createHashMap } from 'zrender/lib/core/util.js';
import makeStyleMapper from '../model/mixin/makeStyleMapper.js';
import { ITEM_STYLE_KEY_MAP } from '../model/mixin/itemStyle.js';
import { LINE_STYLE_KEY_MAP } from '../model/mixin/lineStyle.js';
import Model from '../model/Model.js';
import { makeInner } from '../util/model.js';
var inner = makeInner();
var defaultStyleMappers = {
  itemStyle: makeStyleMapper(ITEM_STYLE_KEY_MAP, true),
  lineStyle: makeStyleMapper(LINE_STYLE_KEY_MAP, true)
};
var defaultColorKey = {
  lineStyle: 'stroke',
  itemStyle: 'fill'
};
function getStyleMapper(seriesModel, stylePath) {
  var styleMapper = seriesModel.visualStyleMapper || defaultStyleMappers[stylePath];
  if (!styleMapper) {
    console.warn("Unknown style type '" + stylePath + "'.");
    return defaultStyleMappers.itemStyle;
  }
  return styleMapper;
}
function getDefaultColorKey(seriesModel, stylePath) {
  // return defaultColorKey[stylePath] ||
  var colorKey = seriesModel.visualDrawType || defaultColorKey[stylePath];
  if (!colorKey) {
    console.warn("Unknown style type '" + stylePath + "'.");
    return 'fill';
  }
  return colorKey;
}
var seriesStyleTask = {
  createOnAllSeries: true,
  performRawSeries: true,
  reset: function (seriesModel, ecModel) {
    var data = seriesModel.getData();
    var stylePath = seriesModel.visualStyleAccessPath || 'itemStyle';
    // Set in itemStyle
    var styleModel = seriesModel.getModel(stylePath);
    var getStyle = getStyleMapper(seriesModel, stylePath);
    var globalStyle = getStyle(styleModel);
    var decalOption = styleModel.getShallow('decal');
    if (decalOption) {
      data.setVisual('decal', decalOption);
      decalOption.dirty = true;
    }
    // TODO
    var colorKey = getDefaultColorKey(seriesModel, stylePath);
    var color = globalStyle[colorKey];
    // TODO style callback
    var colorCallback = isFunction(color) ? color : null;
    var hasAutoColor = globalStyle.fill === 'auto' || globalStyle.stroke === 'auto';
    // Get from color palette by default.
    if (!globalStyle[colorKey] || colorCallback || hasAutoColor) {
      // Note: If some series has color specified (e.g., by itemStyle.color), we DO NOT
      // make it effect palette. Because some scenarios users need to make some series
      // transparent or as background, which should better not effect the palette.
      var colorPalette = seriesModel.getColorFromPalette(
      // TODO series count changed.
      seriesModel.name, null, ecModel.getSeriesCount());
      if (!globalStyle[colorKey]) {
        globalStyle[colorKey] = colorPalette;
        data.setVisual('colorFromPalette', true);
      }
      globalStyle.fill = globalStyle.fill === 'auto' || isFunction(globalStyle.fill) ? colorPalette : globalStyle.fill;
      globalStyle.stroke = globalStyle.stroke === 'auto' || isFunction(globalStyle.stroke) ? colorPalette : globalStyle.stroke;
    }
    data.setVisual('style', globalStyle);
    data.setVisual('drawType', colorKey);
    // Only visible series has each data be visual encoded
    if (!ecModel.isSeriesFiltered(seriesModel) && colorCallback) {
      data.setVisual('colorFromPalette', false);
      return {
        dataEach: function (data, idx) {
          var dataParams = seriesModel.getDataParams(idx);
          var itemStyle = extend({}, globalStyle);
          itemStyle[colorKey] = colorCallback(dataParams);
          data.setItemVisual(idx, 'style', itemStyle);
        }
      };
    }
  }
};
var sharedModel = new Model();
var dataStyleTask = {
  createOnAllSeries: true,
  performRawSeries: true,
  reset: function (seriesModel, ecModel) {
    if (seriesModel.ignoreStyleOnData || ecModel.isSeriesFiltered(seriesModel)) {
      return;
    }
    var data = seriesModel.getData();
    var stylePath = seriesModel.visualStyleAccessPath || 'itemStyle';
    // Set in itemStyle
    var getStyle = getStyleMapper(seriesModel, stylePath);
    var colorKey = data.getVisual('drawType');
    return {
      dataEach: data.hasItemOption ? function (data, idx) {
        // Not use getItemModel for performance considuration
        var rawItem = data.getRawDataItem(idx);
        if (rawItem && rawItem[stylePath]) {
          sharedModel.option = rawItem[stylePath];
          var style = getStyle(sharedModel);
          var existsStyle = data.ensureUniqueItemVisual(idx, 'style');
          extend(existsStyle, style);
          if (sharedModel.option.decal) {
            data.setItemVisual(idx, 'decal', sharedModel.option.decal);
            sharedModel.option.decal.dirty = true;
          }
          if (colorKey in style) {
            data.setItemVisual(idx, 'colorFromPalette', false);
          }
        }
      } : null
    };
  }
};
// Pick color from palette for the data which has not been set with color yet.
// Note: do not support stream rendering. No such cases yet.
var dataColorPaletteTask = {
  performRawSeries: true,
  overallReset: function (ecModel) {
    // Each type of series uses one scope.
    // Pie and funnel are using different scopes.
    var paletteScopeGroupByType = createHashMap();
    ecModel.eachSeries(function (seriesModel) {
      var colorBy = seriesModel.getColorBy();
      if (seriesModel.isColorBySeries()) {
        return;
      }
      var key = seriesModel.type + '-' + colorBy;
      var colorScope = paletteScopeGroupByType.get(key);
      if (!colorScope) {
        colorScope = {};
        paletteScopeGroupByType.set(key, colorScope);
      }
      inner(seriesModel).scope = colorScope;
    });
    ecModel.eachSeries(function (seriesModel) {
      if (seriesModel.isColorBySeries() || ecModel.isSeriesFiltered(seriesModel)) {
        return;
      }
      var dataAll = seriesModel.getRawData();
      var idxMap = {};
      var data = seriesModel.getData();
      var colorScope = inner(seriesModel).scope;
      var stylePath = seriesModel.visualStyleAccessPath || 'itemStyle';
      var colorKey = getDefaultColorKey(seriesModel, stylePath);
      data.each(function (idx) {
        var rawIdx = data.getRawIndex(idx);
        idxMap[rawIdx] = idx;
      });
      // Iterate on data before filtered. To make sure color from palette can be
      // Consistent when toggling legend.
      dataAll.each(function (rawIdx) {
        var idx = idxMap[rawIdx];
        var fromPalette = data.getItemVisual(idx, 'colorFromPalette');
        // Get color from palette for each data only when the color is inherited from series color, which is
        // also picked from color palette. So following situation is not in the case:
        // 1. series.itemStyle.color is set
        // 2. color is encoded by visualMap
        if (fromPalette) {
          var itemStyle = data.ensureUniqueItemVisual(idx, 'style');
          var name_1 = dataAll.getName(rawIdx) || rawIdx + '';
          var dataCount = dataAll.count();
          itemStyle[colorKey] = seriesModel.getColorFromPalette(name_1, colorScope, dataCount);
        }
      });
    });
  }
};
export { seriesStyleTask, dataStyleTask, dataColorPaletteTask };