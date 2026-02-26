import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

text = """(0:03) This is a master course on interior design. (0:06) Anyone can take this course, whether you are in school, college, undergraduate or graduate. (0:11) In this course, you will be taught basic to advanced level of interior design in a residential park.
(0:17) After that, you will have a portfolio based on which you can easily get a job in this profession. (0:23) My motive for making this course is that people who want to come in the interior design profession, (0:27) have not come for some reason, or even if they have come, they have not understood how the actual work is done. (0:34) This is content specially for those people.
(0:37) I myself have gone through this phase. (0:39) I know how it feels to do not know anything about it. (0:43) So this is the content.
(0:44) If you want the content, you can download it from the description. (0:47) You will get the link. (0:48) Let's discuss one by one.
(0:49) Elements and principles of design. (0:51) Because of this, we can make a space aesthetically beautiful. (0:54) In elements, there is line, form, lighting is a very important topic.
(0:59) Then color and pattern are also very important topics. (1:01) Then texture and space. (1:03) In principle, everything is very important.
(1:05) Balance, rhythm and repetition, contrast, focus, unity, scale and proportion. (1:10) What are the styles in interior design? (1:12) There are so many styles that we cannot count. (1:14) But these are some important styles that are important in today's time.
(1:18) Let's see one by one. (1:18) Coastal Chic, Scandinavian, Bohemian, Farmhouse, Contemporary, Industrial, Rustic, Zen, Traditional and Minimalistic. (1:27) Next is the fabric used in interior design.
(1:30) Which fabrics are there and which fabric do we have to choose? (1:33) Like cotton, silk, crayon, nylon, polyester, leather, wool, velvet. (1:38) We will study all this in detail. (1:40) After this, we will pick up a particular space and learn to design it.
(1:44) First, we will learn their technical dimensions. (1:46) Which height is comfortable for what? (1:49) How much width is required for what? (1:50) We will learn their basic dimensions. (1:52) Then we will learn their basic planning.
(1:54) After that, we will learn about materials. (1:56) Which materials can be used where and where? (1:59) Like the kitchen has become separate. (2:00) Living room, bedroom, bathroom.
(2:02) All these have different types of materials. (2:04) Drawings are different. (2:05) We will learn all this.
(2:06) The next major topic is Estimation and Budgeting. (2:09) Then comes software. (2:10) In which AutoCAD, SketchUp and Vray came.
(2:13) In AutoCAD, we will make 2D drawings. (2:16) Which is useful for labor. (2:17) We will see that they make designs.
(2:19) SketchUp and Vray are for 3D. (2:21) In SketchUp, 3D will be made. (2:22) And in Vray, a realistic effect comes.
(2:24) So that the client can know. (2:26) What type of design we are going to do. (2:28) So these are the details of the course.
(0:00) Just like bricks make a building, brick is an element of building. (0:07) Similarly, the things used to make a design are called elements of design. (0:12) First element of design is line.
(0:15) Lines, through which we can make anything. (0:17) When we make any design, many types of lines are used. (0:21) But which line is used the most, that only depicts the sense of our design.
(0:25) There are many types of lines. We have divided them into four types. (0:29) First is direction of line.
(0:31) Horizontal lines are very calm and relaxing. (0:34) If you are designing a space of a person who is very calm in nature. (0:38) For example, if you are designing for people of old age, then you can definitely use these lines.
(0:45) Vertical lines are bold and show the space long. (0:49) You should use these lines where you want to make a distinction. (0:52) For example, if you are designing an office, then you can use these lines where you want to make a boss's place.
(0:58) Diagonal lines add interest to the space. (1:00) If there are only vertical and horizontal lines in the space, then it becomes monotonous and boring. (1:06) But as soon as you add a few diagonal lines, it becomes interesting.
(1:09) Remember not to draw too many diagonal lines. (1:12) And do not draw lines where you want to relax. (1:15) They give motion.
(1:17) The second quality of lines is rhythm. (1:18) Rhythm means when anything is repeated with the same type, it is called rhythm. (1:24) Rhythmic lines are of many types.
(1:26) But let me give you a small tip. (1:28) The edges of the lines are a little masculine. (1:32) And the soft edges, i.e. curves, are a little feminine.
(1:36) Arithmetic lines are made of anything. (1:38) Which looks cluttered. (1:40) It is not used much in its interior.
(1:42) The third quality of line is how thick it is and how thin it is. (1:46) Means weight of line. (1:47) Thin lines are soft and fragile.
(1:50) And thick lines are bold. (1:53) And if it is too thick, it looks angry. (1:56) Thick lines easily gain attention.
(1:58) So we can use it in a specific place where we want to gain attention. (2:02) Next is the style of the line. (2:04) It is casual, formal, funky.
(2:07) It also depends on how the space feels. (2:11) This is a photo of a cafe where more young people come. (2:14) Casual people come.
(2:15) That's why casual fonts are used here. (2:18) But if you look at the next example, where you have to give a bossy feel, (2:22) there you will have to use formal fonts. (2:24) In the next video, we will talk about the second principle of design.
(0:00) Hey, hello everyone, today we are going to talk about shape and form. (0:03) Shape is 2D, like a square, that is shape. (0:07) If we make a cuboid of it, then that is form.
(0:10) So there are two types of shape and form, we can divide it. (0:13) One is organic shapes and forms, which are available in nature. (0:16) Like egg, like any shape of mountains.
(0:20) Random things are available, like tree, leaf, these are all organic shapes. (0:25) But one is inorganic shapes. (0:27) Inorganic shapes are man-made, like a square.
(0:30) You won't find a perfect square in nature. (0:32) You won't find a perfect rectangle in nature. (0:34) You won't even find a perfect circle.
(0:36) So all these things are inorganic shapes. (0:38) If you like organic shapes, everyone has their own taste. (0:43) Like personally, I like inorganic shapes.
(0:46) Maybe you like organic shapes. (0:48) So you have to see what kind of shapes the client likes, you have to design accordingly. (0:53) So you can use these shapes in the headboard.
(0:55) Your rug can be circular, rectangular, anything. (1:00) You can use organic shrugs. (1:02) Your cupboard shape can be anything.
(1:04) Basically, wherever you are designing it, in your furniture pieces, in your rugs, you can use these shapes. (1:11) So use these shapes in a proportion. (1:13) Like if someone likes more organic, then give them more organic.
(1:17) But use inorganic in that too. (1:20) Because we have to make a contrast. (1:21) We have to make a contrast that everything is a little bit.
(1:23) It shouldn't be like that. (1:23) If one has a monotony, then it starts looking boring. (1:26) Like if you add all the organic shapes, then it starts looking boring in itself.
(1:32) So if we add geometric shapes or inorganic shapes, then it will become boring in itself. (1:38) We have to make a mixture of these two. (1:40) One can dominate, the other can be less.
(1:42) But both should be there. (1:44) We can decide these shapes in such a way that (1:46) Like if you are designing a kids room, then you can't give sharp edges. (1:51) You have to give less inorganic shapes.
(1:53) Like the curves of your bed, (1:55) you will see when it meets on the side, (1:58) then we make it a round edge. (2:00) So that it doesn't touch the kids. (2:02) So we have to take care of all these things.
(2:04) Or many styles are like this. (2:05) Like contemporary style. (2:07) In that, you get to see edges.
(2:08) Some traditional styles, you get to see curves. (2:12) We have to take care of all these things. (2:13) Like if the room width is 11 feet or 10 feet.
(2:18) You give a long cupboard in front of it. (2:21) You have to make a niche in between. (2:23) Because if it comes in front of you suddenly, (2:25) then it feels like someone has thrown it on your face.
(2:28) So you have to make a niche in between. (2:30) You have to add some shapes. (2:31) It can be organic or inorganic.
(2:33) You have to give breaks. (2:35) So shapes are used there too. (2:37) That's about it about shapes."""

# 1. Tokenize with NLTK
sentences = sent_tokenize(text)
print(sentences)

#2 chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20)

chunks = text_splitter.create_documents(sentences)
print(chunks[0:4])

#3 Vectorization
import chromadb
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the embedding model (e.g., all-MiniLM-L6-v2)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract text content from Document objects
texts_to_embed = [chunk.page_content for chunk in chunks]

# 2. Create and store embeddings in a vector database (ChromaDB in-memory)
vector_store = Chroma.from_texts(
    texts=texts_to_embed,
    embedding=embedding_model,
    collection_name="video_transcript"
)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(documents = chunks, embedding = OpenAIEmbeddings(model="text-embedding-3-small"))

print(vector_store._collection.get())
retriever = vector_store.as_retriever()

#4 Load model and prompt
from langchain_classic import hub
prompt = hub.pull("rlm/rag-prompt")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
llm.invoke("What is Agent?")

def format_doc(docs):
  return "\n".join(doc.page_content for doc in docs)

#5 Rag Pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
rag_chain = ({"context": retriever| format_doc, "question": RunnablePassthrough()}
             |prompt
             |llm
             |StrOutputParser())

#5Query to LLM
rag_chain.invoke("What's the course about?")
