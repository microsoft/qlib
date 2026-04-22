
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
/**
 * Parse and decode geo json
 */
import * as zrUtil from 'zrender/lib/core/util.js';
import { GeoJSONLineStringGeometry, GeoJSONPolygonGeometry, GeoJSONRegion } from './Region.js';
function decode(json) {
  if (!json.UTF8Encoding) {
    return json;
  }
  var jsonCompressed = json;
  var encodeScale = jsonCompressed.UTF8Scale;
  if (encodeScale == null) {
    encodeScale = 1024;
  }
  var features = jsonCompressed.features;
  zrUtil.each(features, function (feature) {
    var geometry = feature.geometry;
    var encodeOffsets = geometry.encodeOffsets;
    var coordinates = geometry.coordinates;
    // Geometry may be appeded manually in the script after json loaded.
    // In this case this geometry is usually not encoded.
    if (!encodeOffsets) {
      return;
    }
    switch (geometry.type) {
      case 'LineString':
        geometry.coordinates = decodeRing(coordinates, encodeOffsets, encodeScale);
        break;
      case 'Polygon':
        decodeRings(coordinates, encodeOffsets, encodeScale);
        break;
      case 'MultiLineString':
        decodeRings(coordinates, encodeOffsets, encodeScale);
        break;
      case 'MultiPolygon':
        zrUtil.each(coordinates, function (rings, idx) {
          return decodeRings(rings, encodeOffsets[idx], encodeScale);
        });
    }
  });
  // Has been decoded
  jsonCompressed.UTF8Encoding = false;
  return jsonCompressed;
}
function decodeRings(rings, encodeOffsets, encodeScale) {
  for (var c = 0; c < rings.length; c++) {
    rings[c] = decodeRing(rings[c], encodeOffsets[c], encodeScale);
  }
}
function decodeRing(coordinate, encodeOffsets, encodeScale) {
  var result = [];
  var prevX = encodeOffsets[0];
  var prevY = encodeOffsets[1];
  for (var i = 0; i < coordinate.length; i += 2) {
    var x = coordinate.charCodeAt(i) - 64;
    var y = coordinate.charCodeAt(i + 1) - 64;
    // ZigZag decoding
    x = x >> 1 ^ -(x & 1);
    y = y >> 1 ^ -(y & 1);
    // Delta deocding
    x += prevX;
    y += prevY;
    prevX = x;
    prevY = y;
    // Dequantize
    result.push([x / encodeScale, y / encodeScale]);
  }
  return result;
}
export default function parseGeoJSON(geoJson, nameProperty) {
  geoJson = decode(geoJson);
  return zrUtil.map(zrUtil.filter(geoJson.features, function (featureObj) {
    // Output of mapshaper may have geometry null
    return featureObj.geometry && featureObj.properties && featureObj.geometry.coordinates.length > 0;
  }), function (featureObj) {
    var properties = featureObj.properties;
    var geo = featureObj.geometry;
    var geometries = [];
    switch (geo.type) {
      case 'Polygon':
        var coordinates = geo.coordinates;
        // According to the GeoJSON specification.
        // First must be exterior, and the rest are all interior(holes).
        geometries.push(new GeoJSONPolygonGeometry(coordinates[0], coordinates.slice(1)));
        break;
      case 'MultiPolygon':
        zrUtil.each(geo.coordinates, function (item) {
          if (item[0]) {
            geometries.push(new GeoJSONPolygonGeometry(item[0], item.slice(1)));
          }
        });
        break;
      case 'LineString':
        geometries.push(new GeoJSONLineStringGeometry([geo.coordinates]));
        break;
      case 'MultiLineString':
        geometries.push(new GeoJSONLineStringGeometry(geo.coordinates));
    }
    var region = new GeoJSONRegion(properties[nameProperty || 'name'], geometries, properties.cp);
    region.properties = properties;
    return region;
  });
}