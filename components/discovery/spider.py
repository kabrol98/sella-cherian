# scapy has better crawling capabilities than beautifulsoup
import scrapy
import requests
# from scrapy.selector import HtmlXPathSelector

START_URL = 'https://file-examples.com/index.php/sample-documents-download/sample-xls-download/'
GOOG_URL = 'https://www.google.com/search?q=%22.xls%22&lr=&hl=en&as_qdr=all&sxsrf=ALeKk01oARCcYdEkOT2-3Ryf_kPOtC7QqQ:1585698973521&ei=ndiDXpmkH5uztQbDlq6ABw&start=30&sa=N&ved=2ahUKEwjZ5qWA9cXoAhWbWc0KHUOLC3A4FBDw0wN6BAgLED8&biw=1600&bih=800'

class xlsSpider(scrapy.Spider):
    name='xls_spider'
    start_urls = [START_URL]
    
    def parse(self,response):
        XLS_SELECTOR = 'a ::attr(href)'
        
        for data in response.css(XLS_SELECTOR).re('.*\.xlsx$'):
            fname = data[data.rfind('/')+1:]
            sheetURL = data
            res = requests.get(sheetURL)
            with open(f'results/{fname}','wb') as f:
                f.write(res.content)
            yield {
                'url': sheetURL,
                'filename': fname
            }
    
    def google_parse(self, response):
        # hxs = HtmlXPathSelector(response)
        for sel in response.xpath('//div[@id="ires"]//li[@class="g"]//h3[@class="r"]'):
            name = u''.join(sel.xpath(".//text()").extract())
            url = _parse_url(sel.xpath('.//a/@href').extract()[0])
            yield {
                'name':name,
                'url': url
            }
        
        
        # SEARCH_SELECTOR = "//div[@id='ires']"
        
        # for data in response.xpath(SEARCH_SELECTOR).css('a'):
        #     yield {
        #         'data': data.get()
            # }