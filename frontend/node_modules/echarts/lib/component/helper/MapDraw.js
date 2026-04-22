
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
import * as zrUtil from 'zrender/lib/core/util.js';
import RoamController from './RoamController.js';
import * as roamHelper from '../../component/helper/roamHelper.js';
import { onIrrelevantElement } from '../../component/helper/cursorHelper.js';
import * as graphic from '../../util/graphic.js';
import { toggleHoverEmphasis, enableComponentHighDownFeatures, setDefaultStateProxy } from '../../util/states.js';
import geoSourceManager from '../../coord/geo/geoSourceManager.js';
import { getUID } from '../../util/component.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import { getECData } from '../../util/innerStore.js';
import { createOrUpdatePatternFromDecal } from '../../util/decal.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
import { makeInner } from '../../util/model.js';
/**
 * Only these tags enable use `itemStyle` if they are named in SVG.
 * Other tags like <text> <tspan> <image> might not suitable for `itemStyle`.
 * They will not be considered to be styled until some requirements come.
 */
var OPTION_STYLE_ENABLED_TAGS = ['rect', 'circle', 'line', 'ellipse', 'polygon', 'polyline', 'path'];
var OPTION_STYLE_ENABLED_TAG_MAP = zrUtil.createHashMap(OPTION_STYLE_ENABLED_TAGS);
var STATE_TRIGGER_TAG_MAP = zrUtil.createHashMap(OPTION_STYLE_ENABLED_TAGS.concat(['g']));
var LABEL_HOST_MAP = zrUtil.createHashMap(OPTION_STYLE_ENABLED_TAGS.concat(['g']));
var mapLabelRaw = makeInner();
function getFixedItemStyle(model) {
  var itemStyle = model.getItemStyle();
  var areaColor = model.get('areaColor');
  // If user want the color not to be changed when hover,
  // they should both set areaColor and color to be null.
  if (areaColor != null) {
    itemStyle.fill = areaColor;
  }
  return itemStyle;
}
// Only stroke can be used for line.
// Using fill in style if stroke not exits.
// TODO Not sure yet. Perhaps a separate `lineStyle`?
function fixLineStyle(styleHost) {
  var style = styleHost.style;
  if (style) {
    style.stroke = style.stroke || style.fill;
    style.fill = null;
  }
}
var MapDraw = /** @class */function () {
  function MapDraw(api) {
    var group = new graphic.Group();
    this.uid = getUID('ec_map_draw');
    this._controller = new RoamController(api.getZr());
    this._controllerHost = {
      target: group
    };
    this.group = group;
    group.add(this._regionsGroup = new graphic.Group());
    group.add(this._svgGroup = new graphic.Group());
  }
  MapDraw.prototype.draw = function (mapOrGeoModel, ecModel, api, fromView, payload) {
    var isGeo = mapOrGeoModel.mainType === 'geo';
    // Map series has data. GEO model that controlled by map series
    // will be assigned with map data. Other GEO model has no data.
    var data = mapOrGeoModel.getData && mapOrGeoModel.getData();
    isGeo && ecModel.eachComponent({
      mainType: 'series',
      subType: 'map'
    }, function (mapSeries) {
      if (!data && mapSeries.getHostGeoModel() === mapOrGeoModel) {
        data = mapSeries.getData();
      }
    });
    var geo = mapOrGeoModel.coordinateSystem;
    var regionsGroup = this._regionsGroup;
    var group = this.group;
    var transformInfo = geo.getTransformInfo();
    var transformInfoRaw = transformInfo.raw;
    var transformInfoRoam = transformInfo.roam;
    // No animation when first draw or in action
    var isFirstDraw = !regionsGroup.childAt(0) || payload;
    if (isFirstDraw) {
      group.x = transformInfoRoam.x;
      group.y = transformInfoRoam.y;
      group.scaleX = transformInfoRoam.scaleX;
      group.scaleY = transformInfoRoam.scaleY;
      group.dirty();
    } else {
      graphic.updateProps(group, transformInfoRoam, mapOrGeoModel);
    }
    var isVisualEncodedByVisualMap = data && data.getVisual('visualMeta') && data.getVisual('visualMeta').length > 0;
    var viewBuildCtx = {
      api: api,
      geo: geo,
      mapOrGeoModel: mapOrGeoModel,
      data: data,
      isVisualEncodedByVisualMap: isVisualEncodedByVisualMap,
      isGeo: isGeo,
      transformInfoRaw: transformInfoRaw
    };
    if (geo.resourceType === 'geoJSON') {
      this._buildGeoJSON(viewBuildCtx);
    } else if (geo.resourceType === 'geoSVG') {
      this._buildSVG(viewBuildCtx);
    }
    this._updateController(mapOrGeoModel, ecModel, api);
    this._updateMapSelectHandler(mapOrGeoModel, regionsGroup, api, fromView);
  };
  MapDraw.prototype._buildGeoJSON = function (viewBuildCtx) {
    var regionsGroupByName = this._regionsGroupByName = zrUtil.createHashMap();
    var regionsInfoByName = zrUtil.createHashMap();
    var regionsGroup = this._regionsGroup;
    var transformInfoRaw = viewBuildCtx.transformInfoRaw;
    var mapOrGeoModel = viewBuildCtx.mapOrGeoModel;
    var data = viewBuildCtx.data;
    var projection = viewBuildCtx.geo.projection;
    var projectionStream = projection && projection.stream;
    function transformPoint(point, project) {
      if (project) {
        // projection may return null point.
        point = project(point);
      }
      return point && [point[0] * transformInfoRaw.scaleX + transformInfoRaw.x, point[1] * transformInfoRaw.scaleY + transformInfoRaw.y];
    }
    ;
    function transformPolygonPoints(inPoints) {
      var outPoints = [];
      // If projectionStream is provided. Use it instead of single point project.
      var project = !projectionStream && projection && projection.project;
      for (var i = 0; i < inPoints.length; ++i) {
        var newPt = transformPoint(inPoints[i], project);
        newPt && outPoints.push(newPt);
      }
      return outPoints;
    }
    function getPolyShape(points) {
      return {
        shape: {
          points: transformPolygonPoints(points)
        }
      };
    }
    regionsGroup.removeAll();
    // Only when the resource is GeoJSON, there is `geo.regions`.
    zrUtil.each(viewBuildCtx.geo.regions, function (region) {
      var regionName = region.name;
      // Consider in GeoJson properties.name may be duplicated, for example,
      // there is multiple region named "United Kindom" or "France" (so many
      // colonies). And it is not appropriate to merge them in geo, which
      // will make them share the same label and bring trouble in label
      // location calculation.
      var regionGroup = regionsGroupByName.get(regionName);
      var _a = regionsInfoByName.get(regionName) || {},
        dataIdx = _a.dataIdx,
        regionModel = _a.regionModel;
      if (!regionGroup) {
        regionGroup = regionsGroupByName.set(regionName, new graphic.Group());
        regionsGroup.add(regionGroup);
        dataIdx = data ? data.indexOfName(regionName) : null;
        regionModel = viewBuildCtx.isGeo ? mapOrGeoModel.getRegionModel(regionName) : data ? data.getItemModel(dataIdx) : null;
        var silent = regionModel.get('silent', true);
        silent != null && (regionGroup.silent = silent);
        regionsInfoByName.set(regionName, {
          dataIdx: dataIdx,
          regionModel: regionModel
        });
      }
      var polygonSubpaths = [];
      var polylineSubpaths = [];
      zrUtil.each(region.geometries, function (geometry) {
        // Polygon and MultiPolygon
        if (geometry.type === 'polygon') {
          var polys = [geometry.exterior].concat(geometry.interiors || []);
          if (projectionStream) {
            polys = projectPolys(polys, projectionStream);
          }
          zrUtil.each(polys, function (poly) {
            polygonSubpaths.push(new graphic.Polygon(getPolyShape(poly)));
          });
        }
        // LineString and MultiLineString
        else {
          var points = geometry.points;
          if (projectionStream) {
            points = projectPolys(points, projectionStream, true);
          }
          zrUtil.each(points, function (points) {
            polylineSubpaths.push(new graphic.Polyline(getPolyShape(points)));
          });
        }
      });
      var centerPt = transformPoint(region.getCenter(), projection && projection.project);
      function createCompoundPath(subpaths, isLine) {
        if (!subpaths.length) {
          return;
        }
        var compoundPath = new graphic.CompoundPath({
          culling: true,
          segmentIgnoreThreshold: 1,
          shape: {
            paths: subpaths
          }
        });
        regionGroup.add(compoundPath);
        applyOptionStyleForRegion(viewBuildCtx, compoundPath, dataIdx, regionModel);
        resetLabelForRegion(viewBuildCtx, compoundPath, regionName, regionModel, mapOrGeoModel, dataIdx, centerPt);
        if (isLine) {
          fixLineStyle(compoundPath);
          zrUtil.each(compoundPath.states, fixLineStyle);
        }
      }
      createCompoundPath(polygonSubpaths);
      createCompoundPath(polylineSubpaths, true);
    });
    // Ensure children have been added to `regionGroup` before calling them.
    regionsGroupByName.each(function (regionGroup, regionName) {
      var _a = regionsInfoByName.get(regionName),
        dataIdx = _a.dataIdx,
        regionModel = _a.regionModel;
      resetEventTriggerForRegion(viewBuildCtx, regionGroup, regionName, regionModel, mapOrGeoModel, dataIdx);
      resetTooltipForRegion(viewBuildCtx, regionGroup, regionName, regionModel, mapOrGeoModel);
      resetStateTriggerForRegion(viewBuildCtx, regionGroup, regionName, regionModel, mapOrGeoModel);
    }, this);
  };
  MapDraw.prototype._buildSVG = function (viewBuildCtx) {
    var mapName = viewBuildCtx.geo.map;
    var transformInfoRaw = viewBuildCtx.transformInfoRaw;
    this._svgGroup.x = transformInfoRaw.x;
    this._svgGroup.y = transformInfoRaw.y;
    this._svgGroup.scaleX = transformInfoRaw.scaleX;
    this._svgGroup.scaleY = transformInfoRaw.scaleY;
    if (this._svgResourceChanged(mapName)) {
      this._freeSVG();
      this._useSVG(mapName);
    }
    var svgDispatcherMap = this._svgDispatcherMap = zrUtil.createHashMap();
    var focusSelf = false;
    zrUtil.each(this._svgGraphicRecord.named, function (namedItem) {
      // Note that we also allow different elements have the same name.
      // For example, a glyph of a city and the label of the city have
      // the same name and their tooltip info can be defined in a single
      // region option.
      var regionName = namedItem.name;
      var mapOrGeoModel = viewBuildCtx.mapOrGeoModel;
      var data = viewBuildCtx.data;
      var svgNodeTagLower = namedItem.svgNodeTagLower;
      var el = namedItem.el;
      var dataIdx = data ? data.indexOfName(regionName) : null;
      var regionModel = mapOrGeoModel.getRegionModel(regionName);
      if (OPTION_STYLE_ENABLED_TAG_MAP.get(svgNodeTagLower) != null && el instanceof Displayable) {
        applyOptionStyleForRegion(viewBuildCtx, el, dataIdx, regionModel);
      }
      if (el instanceof Displayable) {
        el.culling = true;
      }
      var silent = regionModel.get('silent', true);
      silent != null && (el.silent = silent);
      // We do not know how the SVG like so we'd better not to change z2.
      // Otherwise it might bring some unexpected result. For example,
      // an area hovered that make some inner city can not be clicked.
      el.z2EmphasisLift = 0;
      // If self named:
      if (!namedItem.namedFrom) {
        // label should batter to be displayed based on the center of <g>
        // if it is named rather than displayed on each child.
        if (LABEL_HOST_MAP.get(svgNodeTagLower) != null) {
          resetLabelForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel, dataIdx, null);
        }
        resetEventTriggerForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel, dataIdx);
        resetTooltipForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel);
        if (STATE_TRIGGER_TAG_MAP.get(svgNodeTagLower) != null) {
          var focus_1 = resetStateTriggerForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel);
          if (focus_1 === 'self') {
            focusSelf = true;
          }
          var els = svgDispatcherMap.get(regionName) || svgDispatcherMap.set(regionName, []);
          els.push(el);
        }
      }
    }, this);
    this._enableBlurEntireSVG(focusSelf, viewBuildCtx);
  };
  MapDraw.prototype._enableBlurEntireSVG = function (focusSelf, viewBuildCtx) {
    // It's a little complicated to support blurring the entire geoSVG in series-map.
    // So do not support it until some requirements come.
    // At present, in series-map, only regions can be blurred.
    if (focusSelf && viewBuildCtx.isGeo) {
      var blurStyle = viewBuildCtx.mapOrGeoModel.getModel(['blur', 'itemStyle']).getItemStyle();
      // Only support `opacity` here. Because not sure that other props are suitable for
      // all of the elements generated by SVG (especially for Text/TSpan/Image/... ).
      var opacity_1 = blurStyle.opacity;
      this._svgGraphicRecord.root.traverse(function (el) {
        if (!el.isGroup) {
          // PENDING: clear those settings to SVG elements when `_freeSVG`.
          // (Currently it happen not to be needed.)
          setDefaultStateProxy(el);
          var style = el.ensureState('blur').style || {};
          // Do not overwrite the region style that already set from region option.
          if (style.opacity == null && opacity_1 != null) {
            style.opacity = opacity_1;
          }
          // If `ensureState('blur').style = {}`, there will be default opacity.
          // Enable `stateTransition` (animation).
          el.ensureState('emphasis');
        }
      });
    }
  };
  MapDraw.prototype.remove = function () {
    this._regionsGroup.removeAll();
    this._regionsGroupByName = null;
    this._svgGroup.removeAll();
    this._freeSVG();
    this._controller.dispose();
    this._controllerHost = null;
  };
  MapDraw.prototype.findHighDownDispatchers = function (name, geoModel) {
    if (name == null) {
      return [];
    }
    var geo = geoModel.coordinateSystem;
    if (geo.resourceType === 'geoJSON') {
      var regionsGroupByName = this._regionsGroupByName;
      if (regionsGroupByName) {
        var regionGroup = regionsGroupByName.get(name);
        return regionGroup ? [regionGroup] : [];
      }
    } else if (geo.resourceType === 'geoSVG') {
      return this._svgDispatcherMap && this._svgDispatcherMap.get(name) || [];
    }
  };
  MapDraw.prototype._svgResourceChanged = function (mapName) {
    return this._svgMapName !== mapName;
  };
  MapDraw.prototype._useSVG = function (mapName) {
    var resource = geoSourceManager.getGeoResource(mapName);
    if (resource && resource.type === 'geoSVG') {
      var svgGraphic = resource.useGraphic(this.uid);
      this._svgGroup.add(svgGraphic.root);
      this._svgGraphicRecord = svgGraphic;
      this._svgMapName = mapName;
    }
  };
  MapDraw.prototype._freeSVG = function () {
    var mapName = this._svgMapName;
    if (mapName == null) {
      return;
    }
    var resource = geoSourceManager.getGeoResource(mapName);
    if (resource && resource.type === 'geoSVG') {
      resource.freeGraphic(this.uid);
    }
    this._svgGraphicRecord = null;
    this._svgDispatcherMap = null;
    this._svgGroup.removeAll();
    this._svgMapName = null;
  };
  MapDraw.prototype._updateController = function (mapOrGeoModel, ecModel, api) {
    var geo = mapOrGeoModel.coordinateSystem;
    var controller = this._controller;
    var controllerHost = this._controllerHost;
    // @ts-ignore FIXME:TS
    controllerHost.zoomLimit = mapOrGeoModel.get('scaleLimit');
    controllerHost.zoom = geo.getZoom();
    // roamType is will be set default true if it is null
    // @ts-ignore FIXME:TS
    controller.enable(mapOrGeoModel.get('roam') || false);
    var mainType = mapOrGeoModel.mainType;
    function makeActionBase() {
      var action = {
        type: 'geoRoam',
        componentType: mainType
      };
      action[mainType + 'Id'] = mapOrGeoModel.id;
      return action;
    }
    controller.off('pan').on('pan', function (e) {
      this._mouseDownFlag = false;
      roamHelper.updateViewOnPan(controllerHost, e.dx, e.dy);
      api.dispatchAction(zrUtil.extend(makeActionBase(), {
        dx: e.dx,
        dy: e.dy,
        animation: {
          duration: 0
        }
      }));
    }, this);
    controller.off('zoom').on('zoom', function (e) {
      this._mouseDownFlag = false;
      roamHelper.updateViewOnZoom(controllerHost, e.scale, e.originX, e.originY);
      api.dispatchAction(zrUtil.extend(makeActionBase(), {
        totalZoom: controllerHost.zoom,
        zoom: e.scale,
        originX: e.originX,
        originY: e.originY,
        animation: {
          duration: 0
        }
      }));
    }, this);
    controller.setPointerChecker(function (e, x, y) {
      return geo.containPoint([x, y]) && !onIrrelevantElement(e, api, mapOrGeoModel);
    });
  };
  /**
   * FIXME: this is a temporarily workaround.
   * When `geoRoam` the elements need to be reset in `MapView['render']`, because the props like
   * `ignore` might have been modified by `LabelManager`, and `LabelManager#addLabelsOfSeries`
   * will subsequently cache `defaultAttr` like `ignore`. If do not do this reset, the modified
   * props will have no chance to be restored.
   * Note: This reset should be after `clearStates` in `renderSeries` because `useStates` in
   * `renderSeries` will cache the modified `ignore` to `el._normalState`.
   * TODO:
   * Use clone/immutable in `LabelManager`?
   */
  MapDraw.prototype.resetForLabelLayout = function () {
    this.group.traverse(function (el) {
      var label = el.getTextContent();
      if (label) {
        label.ignore = mapLabelRaw(label).ignore;
      }
    });
  };
  MapDraw.prototype._updateMapSelectHandler = function (mapOrGeoModel, regionsGroup, api, fromView) {
    var mapDraw = this;
    regionsGroup.off('mousedown');
    regionsGroup.off('click');
    // @ts-ignore FIXME:TS resolve type conflict
    if (mapOrGeoModel.get('selectedMode')) {
      regionsGroup.on('mousedown', function () {
        mapDraw._mouseDownFlag = true;
      });
      regionsGroup.on('click', function (e) {
        if (!mapDraw._mouseDownFlag) {
          return;
        }
        mapDraw._mouseDownFlag = false;
      });
    }
  };
  return MapDraw;
}();
;
function applyOptionStyleForRegion(viewBuildCtx, el, dataIndex, regionModel) {
  // All of the path are using `itemStyle`, because
  // (1) Some SVG also use fill on polyline (The different between
  // polyline and polygon is "open" or "close" but not fill or not).
  // (2) For the common props like opacity, if some use itemStyle
  // and some use `lineStyle`, it might confuse users.
  // (3) Most SVG use <path>, where can not detect whether to draw a "line"
  // or a filled shape, so use `itemStyle` for <path>.
  var normalStyleModel = regionModel.getModel('itemStyle');
  var emphasisStyleModel = regionModel.getModel(['emphasis', 'itemStyle']);
  var blurStyleModel = regionModel.getModel(['blur', 'itemStyle']);
  var selectStyleModel = regionModel.getModel(['select', 'itemStyle']);
  // NOTE: DON'T use 'style' in visual when drawing map.
  // This component is used for drawing underlying map for both geo component and map series.
  var normalStyle = getFixedItemStyle(normalStyleModel);
  var emphasisStyle = getFixedItemStyle(emphasisStyleModel);
  var selectStyle = getFixedItemStyle(selectStyleModel);
  var blurStyle = getFixedItemStyle(blurStyleModel);
  // Update the itemStyle if has data visual
  var data = viewBuildCtx.data;
  if (data) {
    // Only visual color of each item will be used. It can be encoded by visualMap
    // But visual color of series is used in symbol drawing
    // Visual color for each series is for the symbol draw
    var style = data.getItemVisual(dataIndex, 'style');
    var decal = data.getItemVisual(dataIndex, 'decal');
    if (viewBuildCtx.isVisualEncodedByVisualMap && style.fill) {
      normalStyle.fill = style.fill;
    }
    if (decal) {
      normalStyle.decal = createOrUpdatePatternFromDecal(decal, viewBuildCtx.api);
    }
  }
  // SVG text, tspan and image can be named but not supporeted
  // to be styled by region option yet.
  el.setStyle(normalStyle);
  el.style.strokeNoScale = true;
  el.ensureState('emphasis').style = emphasisStyle;
  el.ensureState('select').style = selectStyle;
  el.ensureState('blur').style = blurStyle;
  // Enable blur
  setDefaultStateProxy(el);
}
function resetLabelForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel,
// Exist only if `viewBuildCtx.data` exists.
dataIdx,
// If labelXY not provided, use `textConfig.position: 'inside'`
labelXY) {
  var data = viewBuildCtx.data;
  var isGeo = viewBuildCtx.isGeo;
  var isDataNaN = data && isNaN(data.get(data.mapDimension('value'), dataIdx));
  var itemLayout = data && data.getItemLayout(dataIdx);
  // In the following cases label will be drawn
  // 1. In map series and data value is NaN
  // 2. In geo component
  // 3. Region has no series legendIcon, which will be add a showLabel flag in mapSymbolLayout
  if (isGeo || isDataNaN || itemLayout && itemLayout.showLabel) {
    var query = !isGeo ? dataIdx : regionName;
    var labelFetcher = void 0;
    // Consider dataIdx not found.
    if (!data || dataIdx >= 0) {
      labelFetcher = mapOrGeoModel;
    }
    var specifiedTextOpt = labelXY ? {
      normal: {
        align: 'center',
        verticalAlign: 'middle'
      }
    } : null;
    // Caveat: must be called after `setDefaultStateProxy(el);` called.
    // because textContent will be assign with `el.stateProxy` inside.
    setLabelStyle(el, getLabelStatesModels(regionModel), {
      labelFetcher: labelFetcher,
      labelDataIndex: query,
      defaultText: regionName
    }, specifiedTextOpt);
    var textEl = el.getTextContent();
    if (textEl) {
      mapLabelRaw(textEl).ignore = textEl.ignore;
      if (el.textConfig && labelXY) {
        // Compute a relative offset based on the el bounding rect.
        var rect = el.getBoundingRect().clone();
        // Need to make sure the percent position base on the same rect in normal and
        // emphasis state. Otherwise if using boundingRect of el, but the emphasis state
        // has borderWidth (even 0.5px), the text position will be changed obviously
        // if the position is very big like ['1234%', '1345%'].
        el.textConfig.layoutRect = rect;
        el.textConfig.position = [(labelXY[0] - rect.x) / rect.width * 100 + '%', (labelXY[1] - rect.y) / rect.height * 100 + '%'];
      }
    }
    // PENDING:
    // If labelLayout is enabled (test/label-layout.html), el.dataIndex should be specified.
    // But el.dataIndex is also used to determine whether user event should be triggered,
    // where el.seriesIndex or el.dataModel must be specified. At present for a single el
    // there is not case that "only label layout enabled but user event disabled", so here
    // we depends `resetEventTriggerForRegion` to do the job of setting `el.dataIndex`.
    el.disableLabelAnimation = true;
  } else {
    el.removeTextContent();
    el.removeTextConfig();
    el.disableLabelAnimation = null;
  }
}
function resetEventTriggerForRegion(viewBuildCtx, eventTrigger, regionName, regionModel, mapOrGeoModel,
// Exist only if `viewBuildCtx.data` exists.
dataIdx) {
  // setItemGraphicEl, setHoverStyle after all polygons and labels
  // are added to the regionGroup
  if (viewBuildCtx.data) {
    // FIXME: when series-map use a SVG map, and there are duplicated name specified
    // on different SVG elements, after `data.setItemGraphicEl(...)`:
    // (1) all of them will be mounted with `dataIndex`, `seriesIndex`, so that tooltip
    // can be triggered only mouse hover. That's correct.
    // (2) only the last element will be kept in `data`, so that if trigger tooltip
    // by `dispatchAction`, only the last one can be found and triggered. That might be
    // not correct. We will fix it in future if anyone demanding that.
    viewBuildCtx.data.setItemGraphicEl(dataIdx, eventTrigger);
  }
  // series-map will not trigger "geoselectchange" no matter it is
  // based on a declared geo component. Because series-map will
  // trigger "selectchange". If it trigger both the two events,
  // If users call `chart.dispatchAction({type: 'toggleSelect'})`,
  // it not easy to also fire event "geoselectchanged".
  else {
    // Package custom mouse event for geo component
    getECData(eventTrigger).eventData = {
      componentType: 'geo',
      componentIndex: mapOrGeoModel.componentIndex,
      geoIndex: mapOrGeoModel.componentIndex,
      name: regionName,
      region: regionModel && regionModel.option || {}
    };
  }
}
function resetTooltipForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel) {
  if (!viewBuildCtx.data) {
    graphic.setTooltipConfig({
      el: el,
      componentModel: mapOrGeoModel,
      itemName: regionName,
      // @ts-ignore FIXME:TS fix the "compatible with each other"?
      itemTooltipOption: regionModel.get('tooltip')
    });
  }
}
function resetStateTriggerForRegion(viewBuildCtx, el, regionName, regionModel, mapOrGeoModel) {
  // @ts-ignore FIXME:TS fix the "compatible with each other"?
  el.highDownSilentOnTouch = !!mapOrGeoModel.get('selectedMode');
  // @ts-ignore FIXME:TS fix the "compatible with each other"?
  var emphasisModel = regionModel.getModel('emphasis');
  var focus = emphasisModel.get('focus');
  toggleHoverEmphasis(el, focus, emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
  if (viewBuildCtx.isGeo) {
    enableComponentHighDownFeatures(el, mapOrGeoModel, regionName);
  }
  return focus;
}
function projectPolys(rings,
// Polygons include exterior and interiors. Or polylines.
createStream, isLine) {
  var polygons = [];
  var curPoly;
  function startPolygon() {
    curPoly = [];
  }
  function endPolygon() {
    if (curPoly.length) {
      polygons.push(curPoly);
      curPoly = [];
    }
  }
  var stream = createStream({
    polygonStart: startPolygon,
    polygonEnd: endPolygon,
    lineStart: startPolygon,
    lineEnd: endPolygon,
    point: function (x, y) {
      // May have NaN values from stream.
      if (isFinite(x) && isFinite(y)) {
        curPoly.push([x, y]);
      }
    },
    sphere: function () {}
  });
  !isLine && stream.polygonStart();
  zrUtil.each(rings, function (ring) {
    stream.lineStart();
    for (var i = 0; i < ring.length; i++) {
      stream.point(ring[i][0], ring[i][1]);
    }
    stream.lineEnd();
  });
  !isLine && stream.polygonEnd();
  return polygons;
}
export default MapDraw;
// @ts-ignore FIXME:TS fix the "compatible with each other"?