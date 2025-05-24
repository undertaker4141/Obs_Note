# 題目
![[lab_14-1748080643253.png|1116x533]]

|                                         |                                        |
| --------------------------------------- | -------------------------------------- |
| ![[lab_14-1748081139135.png\|1000x464]] | ![[lab_14-1748081150024.png\|834x464]] |


# 結果圖
## 初始化 welcome 訊息

|                               |                               |
| ----------------------------- | ----------------------------- |
| ![[lab_14-1748080987978.png]] | ![[lab_14-1748081267858.png]] |
- ### 初始 welcome 執行了三次，可能是 **MCU 因不明因素重置** ，code 部分應該是無誤，我有嘗試調整設定但依舊無法解決此問題。
   

## help 指令

|                                         |                                        |
| --------------------------------------- | -------------------------------------- |
| ![[lab_14-1748081356970.png\|1003x574]] | ![[lab_14-1748081380364.png\|725x437]] |

   

## info 指令

|                               |                                        |
| ----------------------------- | -------------------------------------- |
| ![[lab_14-1748081474515.png]] | ![[lab_14-1748081495805.png\|727x438]] |
|                               |                                        |

   

## pc13 high 指令

|                               |                               |
| ----------------------------- | ----------------------------- |
| ![[lab_14-1748081564348.png]] | ![[lab_14-1748084520766.png]] |
| ![[lab_14-1748085005051.png]] |                               |
## pc13 low

|                               |                               |
| ----------------------------- | ----------------------------- |
| ![[lab_14-1748085075876.png]] | ![[lab_14-1748085109274.png]] |
| ![[lab_14-1748085185324.png]] |                               |
## 非 command 字串

|                               |                               |
| ----------------------------- | ----------------------------- |
| ![[lab_14-1748085264051.png]] | ![[lab_14-1748085302515.png]] |
# 程式碼
```c
#include <stm32f10x.h>
#include <string.h>

char usart1_recByte();
void usart1_sendByte(char c);
void usart1_sendStr(char *str); //字串輸出
void delay_ms(uint16_t t);

int main() {
	/*設定UART及LED基本參數*/
    RCC->APB2ENR |= (1 << 2);
    RCC->APB2ENR |= (1 << 14);
    RCC->APB2ENR |= (1 << 4);

    GPIOA->CRH |= 0x000008B0;
    GPIOA->ODR |= (1 << 10);

    USART1->BRR = 7500;
    USART1->CR1 = 0x200C;

    GPIOC->CRH = 0x44344444; //PC13 output mode

    /*清除初始垃圾數據(經測試會因為電腦及設定不同而有不同的結果，可先註解試試是否第一次指令會讀到<0xff>的dummy值再決定是否調整或啟用這段程式碼)*/
    delay_ms(100);
    if(USART1->SR & (1 << 5)){
    	usart1_recByte();
    }

    /*初始化歡迎語*/
    char welcome[] = "Welcome to our command line Interface!\r\nYou can control Hardware by using commands\r\n";
    usart1_sendStr(welcome);

    /*cmd_line 迴圈*/
    char cmd_line[] = "cmd>>";
    char help[] = "The commands you can use:\r\ninfo:Show the information about the processor.\r\npc13 high:Make PC13 output HIGH.\r\npc13 low:Make PC13 output LOW.\r\n";
    char info[] = "Hello, You are using STM32F103C8.\r\nARM Cortex M3 CPU.\r\n72MHz max freq\r\n64KB of flash memory and 20 KB of SRAM\r\n";
    char pc13_high[] = "PC13 set to HIGH\r\n";
    char pc13_low[] = "PC13 set to LOW\r\n";
    char unknown[] = "Unknown command.\r\n";
    while(1) {
    	usart1_sendStr(cmd_line);

    	/*讀取使用者輸入*/
    	char user_c[105]; //假定使用者輸入不超過100字元
    	for(int i = 0;i<105;i++){
    		user_c[i] = '\0';
    	}
    	int ccout = 0;
    	while(1){
    		user_c[ccout] = usart1_recByte();
    		if(user_c[ccout] == '\r')break;
    		ccout++;
    	}


    	user_c[ccout] = '\0';
    	usart1_sendStr(user_c);
    	usart1_sendStr("\r\n");

    	/*cmd邏輯*/
    	if(strcmp(user_c,"help")==0){
    		usart1_sendStr(help);
    	}
    	else if(strcmp(user_c,"info")==0){
    		usart1_sendStr(info);
    	}
    	else if(strcmp(user_c,"pc13 high")==0){
    		usart1_sendStr(pc13_high);
    		GPIOC->BSRR = (1 << 13);
    	}
    	else if(strcmp(user_c,"pc13 low")==0){
    	    usart1_sendStr(pc13_low);
    	    GPIOC->BRR = (1 << 13);
    	}
    	else{
    		usart1_sendStr(unknown);
    	}
    }
}

char usart1_recByte()
{
    while ((USART1->SR & (1 << 5)) == 0);
    return USART1->DR;
}

void usart1_sendByte(char c) {
    USART1->DR = c;
    while ((USART1->SR & (1 << 6)) == 0);
}

void usart1_sendStr(char *str) {
	int counter = 0;
	while(str[counter] != '\0'){
			usart1_sendByte(str[counter]);
			counter++;
		}
	}

//The following delay is tested with PICSimlab and 62.5MHz
void delay_ms(uint16_t t)
{
	volatile unsigned long l = 0;

	for(uint16_t i = 0; i < t; i++)
		for(l = 0; l < 9000; l++) ;
}
```
