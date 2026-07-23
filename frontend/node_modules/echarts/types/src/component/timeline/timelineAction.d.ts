import { Payload } from '../../util/types.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export interface TimelineChangePayload extends Payload {
    type: 'timelineChange';
    currentIndex: number;
}
export interface TimelinePlayChangePayload extends Payload {
    type: 'timelinePlayChange';
    playState: boolean;
}
export declare function installTimelineAction(registers: EChartsExtensionInstallRegisters): void;
