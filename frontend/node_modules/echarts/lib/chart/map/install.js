
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
import { use } from '../../extension.js';
import MapView from './MapView.js';
import MapSeries from './MapSeries.js';
import mapDataStatistic from './mapDataStatistic.js';
import mapSymbolLayout from './mapSymbolLayout.js';
import { createLegacyDataSelectAction } from '../../legacy/dataSelectAction.js';
import { install as installGeo } from '../../component/geo/install.js';
export function install(registers) {
  use(installGeo);
  registers.registerChartView(MapView);
  registers.registerSeriesModel(MapSeries);
  registers.registerLayout(mapSymbolLayout);
  registers.registerProcessor(registers.PRIORITY.PROCESSOR.STATISTIC, mapDataStatistic);
  createLegacyDataSelectAction('map', registers.registerAction);
}