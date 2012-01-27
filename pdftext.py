from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice

class PDFText:
	
	def __init__(self, filepath):
		self.doc = PDFDocument() # the underlying pdf document
		fp = open(filepath, 'rb')
		parser = PDFParser(fp)
		parser.set_document(self.doc)
		self.doc.set_parser(parser)
		self.doc.initialize()
		
	def words(self):
		return []