
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
// In somehow. If we export like
// export * as LineChart './chart/line/install'
// The exported code will be transformed to
// import * as LineChart_1 './chart/line/install'; export {LineChart_1 as LineChart};
// Treeshaking in webpack will not work even if we configured sideEffects to false in package.json
export { install as LineChart } from '../chart/line/install.js';
export { install as BarChart } from '../chart/bar/install.js';
export { install as PieChart } from '../chart/pie/install.js';
export { install as ScatterChart } from '../chart/scatter/install.js';
export { install as RadarChart } from '../chart/radar/install.js';
export { install as MapChart } from '../chart/map/install.js';
export { install as TreeChart } from '../chart/tree/install.js';
export { install as TreemapChart } from '../chart/treemap/install.js';
export { install as GraphChart } from '../chart/graph/install.js';
export { install as GaugeChart } from '../chart/gauge/install.js';
export { install as FunnelChart } from '../chart/funnel/install.js';
export { install as ParallelChart } from '../chart/parallel/install.js';
export { install as SankeyChart } from '../chart/sankey/install.js';
export { install as BoxplotChart } from '../chart/boxplot/install.js';
export { install as CandlestickChart } from '../chart/candlestick/install.js';
export { install as EffectScatterChart } from '../chart/effectScatter/install.js';
export { install as LinesChart } from '../chart/lines/install.js';
export { install as HeatmapChart } from '../chart/heatmap/install.js';
export { install as PictorialBarChart } from '../chart/bar/installPictorialBar.js';
export { install as ThemeRiverChart } from '../chart/themeRiver/install.js';
export { install as SunburstChart } from '../chart/sunburst/install.js';
export { install as CustomChart } from '../chart/custom/install.js';