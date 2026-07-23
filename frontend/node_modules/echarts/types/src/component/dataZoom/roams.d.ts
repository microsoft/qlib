import ExtensionAPI from '../../core/ExtensionAPI.js';
import InsideZoomModel from './InsideZoomModel.js';
import { DataZoomGetRangeHandlers } from './InsideZoomView.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export declare function setViewInfoToCoordSysRecord(api: ExtensionAPI, dataZoomModel: InsideZoomModel, getRange: DataZoomGetRangeHandlers): void;
export declare function disposeCoordSysRecordIfNeeded(api: ExtensionAPI, dataZoomModel: InsideZoomModel): void;
export declare function installDataZoomRoamProcessor(registers: EChartsExtensionInstallRegisters): void;
