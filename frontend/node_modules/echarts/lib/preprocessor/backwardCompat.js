
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
import { each, isArray, isObject, isTypedArray, defaults } from 'zrender/lib/core/util.js';
import compatStyle from './helper/compatStyle.js';
import { normalizeToArray } from '../util/model.js';
import { deprecateLog, deprecateReplaceLog } from '../util/log.js';
function get(opt, path) {
  var pathArr = path.split(',');
  var obj = opt;
  for (var i = 0; i < pathArr.length; i++) {
    obj = obj && obj[pathArr[i]];
    if (obj == null) {
      break;
    }
  }
  return obj;
}
function set(opt, path, val, overwrite) {
  var pathArr = path.split(',');
  var obj = opt;
  var key;
  var i = 0;
  for (; i < pathArr.length - 1; i++) {
    key = pathArr[i];
    if (obj[key] == null) {
      obj[key] = {};
    }
    obj = obj[key];
  }
  if (overwrite || obj[pathArr[i]] == null) {
    obj[pathArr[i]] = val;
  }
}
function compatLayoutProperties(option) {
  option && each(LAYOUT_PROPERTIES, function (prop) {
    if (prop[0] in option && !(prop[1] in option)) {
      option[prop[1]] = option[prop[0]];
    }
  });
}
var LAYOUT_PROPERTIES = [['x', 'left'], ['y', 'top'], ['x2', 'right'], ['y2', 'bottom']];
var COMPATITABLE_COMPONENTS = ['grid', 'geo', 'parallel', 'legend', 'toolbox', 'title', 'visualMap', 'dataZoom', 'timeline'];
var BAR_ITEM_STYLE_MAP = [['borderRadius', 'barBorderRadius'], ['borderColor', 'barBorderColor'], ['borderWidth', 'barBorderWidth']];
function compatBarItemStyle(option) {
  var itemStyle = option && option.itemStyle;
  if (itemStyle) {
    for (var i = 0; i < BAR_ITEM_STYLE_MAP.length; i++) {
      var oldName = BAR_ITEM_STYLE_MAP[i][1];
      var newName = BAR_ITEM_STYLE_MAP[i][0];
      if (itemStyle[oldName] != null) {
        itemStyle[newName] = itemStyle[oldName];
        if (process.env.NODE_ENV !== 'production') {
          deprecateReplaceLog(oldName, newName);
        }
      }
    }
  }
}
function compatPieLabel(option) {
  if (!option) {
    return;
  }
  if (option.alignTo === 'edge' && option.margin != null && option.edgeDistance == null) {
    if (process.env.NODE_ENV !== 'production') {
      deprecateReplaceLog('label.margin', 'label.edgeDistance', 'pie');
    }
    option.edgeDistance = option.margin;
  }
}
function compatSunburstState(option) {
  if (!option) {
    return;
  }
  if (option.downplay && !option.blur) {
    option.blur = option.downplay;
    if (process.env.NODE_ENV !== 'production') {
      deprecateReplaceLog('downplay', 'blur', 'sunburst');
    }
  }
}
function compatGraphFocus(option) {
  if (!option) {
    return;
  }
  if (option.focusNodeAdjacency != null) {
    option.emphasis = option.emphasis || {};
    if (option.emphasis.focus == null) {
      if (process.env.NODE_ENV !== 'production') {
        deprecateReplaceLog('focusNodeAdjacency', 'emphasis: { focus: \'adjacency\'}', 'graph/sankey');
      }
      option.emphasis.focus = 'adjacency';
    }
  }
}
function traverseTree(data, cb) {
  if (data) {
    for (var i = 0; i < data.length; i++) {
      cb(data[i]);
      data[i] && traverseTree(data[i].children, cb);
    }
  }
}
export default function globalBackwardCompat(option, isTheme) {
  compatStyle(option, isTheme);
  // Make sure series array for model initialization.
  option.series = normalizeToArray(option.series);
  each(option.series, function (seriesOpt) {
    if (!isObject(seriesOpt)) {
      return;
    }
    var seriesType = seriesOpt.type;
    if (seriesType === 'line') {
      if (seriesOpt.clipOverflow != null) {
        seriesOpt.clip = seriesOpt.clipOverflow;
        if (process.env.NODE_ENV !== 'production') {
          deprecateReplaceLog('clipOverflow', 'clip', 'line');
        }
      }
    } else if (seriesType === 'pie' || seriesType === 'gauge') {
      if (seriesOpt.clockWise != null) {
        seriesOpt.clockwise = seriesOpt.clockWise;
        if (process.env.NODE_ENV !== 'production') {
          deprecateReplaceLog('clockWise', 'clockwise');
        }
      }
      compatPieLabel(seriesOpt.label);
      var data = seriesOpt.data;
      if (data && !isTypedArray(data)) {
        for (var i = 0; i < data.length; i++) {
          compatPieLabel(data[i]);
        }
      }
      if (seriesOpt.hoverOffset != null) {
        seriesOpt.emphasis = seriesOpt.emphasis || {};
        if (seriesOpt.emphasis.scaleSize = null) {
          if (process.env.NODE_ENV !== 'production') {
            deprecateReplaceLog('hoverOffset', 'emphasis.scaleSize');
          }
          seriesOpt.emphasis.scaleSize = seriesOpt.hoverOffset;
        }
      }
    } else if (seriesType === 'gauge') {
      var pointerColor = get(seriesOpt, 'pointer.color');
      pointerColor != null && set(seriesOpt, 'itemStyle.color', pointerColor);
    } else if (seriesType === 'bar') {
      compatBarItemStyle(seriesOpt);
      compatBarItemStyle(seriesOpt.backgroundStyle);
      compatBarItemStyle(seriesOpt.emphasis);
      var data = seriesOpt.data;
      if (data && !isTypedArray(data)) {
        for (var i = 0; i < data.length; i++) {
          if (typeof data[i] === 'object') {
            compatBarItemStyle(data[i]);
            compatBarItemStyle(data[i] && data[i].emphasis);
          }
        }
      }
    } else if (seriesType === 'sunburst') {
      var highlightPolicy = seriesOpt.highlightPolicy;
      if (highlightPolicy) {
        seriesOpt.emphasis = seriesOpt.emphasis || {};
        if (!seriesOpt.emphasis.focus) {
          seriesOpt.emphasis.focus = highlightPolicy;
          if (process.env.NODE_ENV !== 'production') {
            deprecateReplaceLog('highlightPolicy', 'emphasis.focus', 'sunburst');
          }
        }
      }
      compatSunburstState(seriesOpt);
      traverseTree(seriesOpt.data, compatSunburstState);
    } else if (seriesType === 'graph' || seriesType === 'sankey') {
      compatGraphFocus(seriesOpt);
      // TODO nodes, edges?
    } else if (seriesType === 'map') {
      if (seriesOpt.mapType && !seriesOpt.map) {
        if (process.env.NODE_ENV !== 'production') {
          deprecateReplaceLog('mapType', 'map', 'map');
        }
        seriesOpt.map = seriesOpt.mapType;
      }
      if (seriesOpt.mapLocation) {
        if (process.env.NODE_ENV !== 'production') {
          deprecateLog('`mapLocation` is not used anymore.');
        }
        defaults(seriesOpt, seriesOpt.mapLocation);
      }
    }
    if (seriesOpt.hoverAnimation != null) {
      seriesOpt.emphasis = seriesOpt.emphasis || {};
      if (seriesOpt.emphasis && seriesOpt.emphasis.scale == null) {
        if (process.env.NODE_ENV !== 'production') {
          deprecateReplaceLog('hoverAnimation', 'emphasis.scale');
        }
        seriesOpt.emphasis.scale = seriesOpt.hoverAnimation;
      }
    }
    compatLayoutProperties(seriesOpt);
  });
  // dataRange has changed to visualMap
  if (option.dataRange) {
    option.visualMap = option.dataRange;
  }
  each(COMPATITABLE_COMPONENTS, function (componentName) {
    var options = option[componentName];
    if (options) {
      if (!isArray(options)) {
        options = [options];
      }
      each(options, function (option) {
        compatLayoutProperties(option);
      });
    }
  });
}