import datetime


class Post:
    """
    Represents a forum post, including contents, meta-infromation and relation to other posts in the thread.
    """
    
    def __init__(self, post_id: int, thread_id: int, author: str, contents: str, 
                 date: datetime.date, reply_to: int=None):
        """
        :param post_id: (int) The unique identifier for the post.
        :param thread_id : (int) The identifier of the thread to which the post belongs.
        :param author : (str) The author of the post.
        :param contents : (str) The content of the post.
        :param date : (date) The date and time when the post was created.
        :param reply_to : (int, optional) The identifier of the post being replied to, 
        None if first post in the thread.
        """
        self.id = post_id
        self.thread_id = thread_id
        self.author = author
        self.contents = contents
        self.date = date
        self.reply_to = reply_to
        
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return self.id
            
    def __str__(self) -> str:
        return f"Post {self.id} in thread {self.thread_id} by {self.author}, contents: '{self.contents[:50]}'"
    
    

class Thread:
    """
    Represents a forum thread. Contains meta-information as well as the thread's posts.
    """
    
    def __init__(self, thread_id, title, author):
        """
        Create a new Thread container.
        :param thread_id: The unique identifier for the thread.
        :param title: The title of the thread.
        :param author: The author of the thread.
        """
        self.id = thread_id
        self.title = title
        self.author = author
        self.posts = []
        
    def add_post(self, post: Post) -> None:
        """
        Add a post to the Thread.
        :param post: the post to be added
        """
        self.posts.append(post)
        
    def __str__(self) -> str:
        return f"Thread {self.id} by {self.author}, titled: '{self.title}'"