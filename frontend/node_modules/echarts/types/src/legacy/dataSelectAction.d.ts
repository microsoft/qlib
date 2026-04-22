import Eventful from 'zrender/lib/core/Eventful.js';
import type { EChartsType, registerAction } from '../core/echarts.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
export declare function createLegacyDataSelectAction(seriesType: string, ecRegisterAction: typeof registerAction): void;
export declare function handleLegacySelectEvents(messageCenter: Eventful, ecIns: EChartsType, api: ExtensionAPI): void;
