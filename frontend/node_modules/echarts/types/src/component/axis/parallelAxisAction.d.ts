import { Payload } from '../../util/types.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export interface ParallelAxisExpandPayload extends Payload {
    axisExpandWindow?: number[];
}
export declare function installParallelActions(registers: EChartsExtensionInstallRegisters): void;
