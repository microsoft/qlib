
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
import { use } from './lib/extension.js';
export * from './lib/export/core.js';
import { install as CanvasRenderer } from './lib/renderer/installCanvasRenderer.js';
import { install as LineChart } from './lib/chart/line/install.js';
import { install as BarChart } from './lib/chart/bar/install.js';
import { install as PieChart } from './lib/chart/pie/install.js';
import { install as GridSimpleComponent } from './lib/component/grid/installSimple.js';
import { install as AriaComponent } from './lib/component/aria/install.js';
import { install as DatasetComponent } from './lib/component/dataset/install.js';
use([CanvasRenderer]);
use([LineChart, BarChart, PieChart]);
use([GridSimpleComponent, AriaComponent, DatasetComponent]);