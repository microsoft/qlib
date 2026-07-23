
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
import SeriesData from '../../data/SeriesData.js';
import Graph from '../../data/Graph.js';
import linkSeriesData from '../../data/helper/linkSeriesData.js';
import prepareSeriesDataSchema from '../../data/helper/createDimensions.js';
import CoordinateSystem from '../../core/CoordinateSystem.js';
import createSeriesData from './createSeriesData.js';
import { convertOptionIdName } from '../../util/model.js';
export default function createGraphFromNodeEdge(nodes, edges, seriesModel, directed, beforeLink) {
  // ??? TODO
  // support dataset?
  var graph = new Graph(directed);
  for (var i = 0; i < nodes.length; i++) {
    graph.addNode(zrUtil.retrieve(
    // Id, name, dataIndex
    nodes[i].id, nodes[i].name, i), i);
  }
  var linkNameList = [];
  var validEdges = [];
  var linkCount = 0;
  for (var i = 0; i < edges.length; i++) {
    var link = edges[i];
    var source = link.source;
    var target = link.target;
    // addEdge may fail when source or target not exists
    if (graph.addEdge(source, target, linkCount)) {
      validEdges.push(link);
      linkNameList.push(zrUtil.retrieve(convertOptionIdName(link.id, null), source + ' > ' + target));
      linkCount++;
    }
  }
  var coordSys = seriesModel.get('coordinateSystem');
  var nodeData;
  if (coordSys === 'cartesian2d' || coordSys === 'polar') {
    nodeData = createSeriesData(nodes, seriesModel);
  } else {
    var coordSysCtor = CoordinateSystem.get(coordSys);
    var coordDimensions = coordSysCtor ? coordSysCtor.dimensions || [] : [];
    // FIXME: Some geo do not need `value` dimenson, whereas `calendar` needs
    // `value` dimension, but graph need `value` dimension. It's better to
    // uniform this behavior.
    if (zrUtil.indexOf(coordDimensions, 'value') < 0) {
      coordDimensions.concat(['value']);
    }
    var dimensions = prepareSeriesDataSchema(nodes, {
      coordDimensions: coordDimensions,
      encodeDefine: seriesModel.getEncode()
    }).dimensions;
    nodeData = new SeriesData(dimensions, seriesModel);
    nodeData.initData(nodes);
  }
  var edgeData = new SeriesData(['value'], seriesModel);
  edgeData.initData(validEdges, linkNameList);
  beforeLink && beforeLink(nodeData, edgeData);
  linkSeriesData({
    mainData: nodeData,
    struct: graph,
    structAttr: 'graph',
    datas: {
      node: nodeData,
      edge: edgeData
    },
    datasAttr: {
      node: 'data',
      edge: 'edgeData'
    }
  });
  // Update dataIndex of nodes and edges because invalid edge may be removed
  graph.update();
  return graph;
}